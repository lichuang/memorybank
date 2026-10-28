//! P1-3: bank creation plus tx0 (`create_source`).

use engine::Error;
use engine::ports::Store;
use engine::types::{Bank, BankId, BankProfile, Source, SourceId, SourceStatus, Timestamp};
use store::SqliteStore;
use store::testing::memory_pool;

const NOW_MILLIS: i64 = 1_700_000_000_123;

fn now() -> Timestamp {
  Timestamp::from_timestamp_millis(NOW_MILLIS).expect("valid timestamp")
}

fn profile() -> BankProfile {
  BankProfile {
    background: "I keep track of people I meet.".into(),
    skepticism: 3,
    literalism: 2,
    empathy: 4,
    bias_strength: 0.5,
  }
}

fn bank(name: &str) -> Bank {
  Bank {
    id: BankId::new(),
    name: name.into(),
    profile: profile(),
    created_at: now(),
    updated_at: now(),
  }
}

fn source(bank_id: BankId, payload: &str) -> Source {
  Source {
    id: SourceId::new(),
    bank_id,
    title: Some("chat".into()),
    payload_text: payload.into(),
    status: SourceStatus::Received,
    created_at: now(),
    updated_at: now(),
  }
}

async fn store() -> SqliteStore {
  SqliteStore::new(memory_pool().await.expect("in-memory store"))
}

#[tokio::test]
async fn bank_round_trips_through_the_port() {
  let store = store().await;
  let bank = bank("primary");

  store.create_bank(bank.clone()).await.expect("create bank");
  let loaded = store.load_bank(bank.id).await.expect("load bank");

  assert_eq!(loaded, bank);
}

#[tokio::test]
async fn missing_bank_is_not_found() {
  let store = store().await;

  let error = store
    .load_bank(BankId::new())
    .await
    .expect_err("missing bank");

  assert!(matches!(error, Error::NotFound("bank", _)));
  assert!(!error.is_retryable());
}

#[tokio::test]
async fn duplicate_bank_name_is_conflict_and_keeps_the_first_row() {
  let store = store().await;
  let first = bank("primary");
  store.create_bank(first.clone()).await.expect("first bank");

  let second = bank("primary");
  let error = store
    .create_bank(second.clone())
    .await
    .expect_err("duplicate name");

  assert!(matches!(error, Error::Conflict("bank", _)));
  assert!(!error.is_retryable());
  assert_eq!(
    store.load_bank(first.id).await.expect("first intact"),
    first
  );
  assert!(
    store.load_bank(second.id).await.is_err(),
    "the rejected bank leaves no row"
  );
}

#[tokio::test]
async fn source_is_persisted_with_received_status() {
  let store = store().await;
  let bank = bank("primary");
  store.create_bank(bank.clone()).await.expect("create bank");

  let source = source(bank.id, "hello\nworld");
  store
    .create_source(source.clone())
    .await
    .expect("create source");

  type Row = (String, String, Option<String>, String, String, i64, i64);
  let row = sqlx::query_as::<_, Row>(
    "SELECT id, bank_id, title, payload_text, status, created_at, updated_at \
     FROM source WHERE id = ?",
  )
  .bind(source.id.to_string())
  .fetch_one(store.pool())
  .await
  .expect("read back source");

  assert_eq!(row.0, source.id.to_string());
  assert_eq!(row.1, bank.id.to_string());
  assert_eq!(row.2.as_deref(), Some("chat"));
  assert_eq!(row.3, "hello\nworld");
  assert_eq!(row.4, "received");
  assert_eq!(row.5, NOW_MILLIS);
  assert_eq!(row.6, NOW_MILLIS);
}

#[tokio::test]
async fn source_status_chunked_is_persisted() {
  let store = store().await;
  let bank = bank("primary");
  store.create_bank(bank.clone()).await.expect("create bank");

  let mut source = source(bank.id, "done");
  source.status = SourceStatus::Chunked;
  let id = source.id;
  store.create_source(source).await.expect("create source");

  let status: String = sqlx::query_scalar("SELECT status FROM source WHERE id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("read status");

  assert_eq!(status, "chunked");
}

#[tokio::test]
async fn duplicate_source_id_is_conflict_and_keeps_one_row() {
  let store = store().await;
  let bank = bank("primary");
  store.create_bank(bank.clone()).await.expect("create bank");

  let source = source(bank.id, "payload");
  store
    .create_source(source.clone())
    .await
    .expect("first source");

  let error = store.create_source(source).await.expect_err("duplicate id");

  assert!(matches!(error, Error::Conflict("source", _)));
  let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM source")
    .fetch_one(store.pool())
    .await
    .expect("count sources");
  assert_eq!(count, 1, "no half-write after the failure");
}

#[tokio::test]
async fn duplicate_callers_each_get_a_new_row() {
  let store = store().await;
  let bank = bank("primary");
  store.create_bank(bank.clone()).await.expect("create bank");

  let first = source(bank.id, "same payload");
  let second = source(bank.id, "same payload");
  store.create_source(first.clone()).await.expect("first");
  store.create_source(second.clone()).await.expect("second");

  let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM source")
    .fetch_one(store.pool())
    .await
    .expect("count sources");

  assert_eq!(count, 2, "the source layer does not deduplicate");
  assert_ne!(first.id, second.id);
}
