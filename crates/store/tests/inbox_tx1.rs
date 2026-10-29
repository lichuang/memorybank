//! P1-4: tx1 (`append_chunks`) — conflict skipping and status advance.

use engine::Error;
use engine::ports::{NewChunk, Store};
use engine::testing::timestamp;
use engine::types::{Bank, BankId, BankProfile, ChunkId, Source, SourceId, SourceStatus};
use store::SqliteStore;
use store::testing::memory_pool;

const SOURCE_TIME: i64 = 1_000;

fn bank() -> Bank {
  Bank {
    id: BankId::new(),
    name: "primary".into(),
    profile: BankProfile {
      background: "background".into(),
      skepticism: 3,
      literalism: 3,
      empathy: 3,
      bias_strength: 0.5,
    },
    created_at: timestamp(SOURCE_TIME),
    updated_at: timestamp(SOURCE_TIME),
  }
}

fn source(bank_id: BankId, payload: &str) -> Source {
  Source {
    id: SourceId::new(),
    bank_id,
    title: None,
    payload_text: payload.into(),
    status: SourceStatus::Received,
    created_at: timestamp(SOURCE_TIME),
    updated_at: timestamp(SOURCE_TIME),
  }
}

fn chunk(
  bank_id: BankId,
  source_id: SourceId,
  index: u32,
  hash: &str,
  created_at: i64,
) -> NewChunk {
  NewChunk {
    id: ChunkId::new(),
    bank_id,
    source_id,
    chunk_index: index,
    text: format!("text {index}"),
    content_hash: hash.into(),
    time_start: None,
    time_end: None,
    created_at: timestamp(created_at),
  }
}

async fn setup() -> (SqliteStore, Bank, Source) {
  let store = SqliteStore::new(memory_pool().await.expect("in-memory store"));
  let bank = bank();
  store.create_bank(bank.clone()).await.expect("create bank");
  let source = source(bank.id, "payload");
  store
    .create_source(source.clone())
    .await
    .expect("create source");
  (store, bank, source)
}

async fn source_status(store: &SqliteStore, source: SourceId) -> (String, i64) {
  sqlx::query_as("SELECT status, updated_at FROM source WHERE id = ?")
    .bind(source.to_string())
    .fetch_one(store.pool())
    .await
    .expect("source row")
}

async fn chunk_count(store: &SqliteStore, bank: BankId) -> i64 {
  sqlx::query_scalar("SELECT COUNT(*) FROM chunk WHERE bank_id = ?")
    .bind(bank.to_string())
    .fetch_one(store.pool())
    .await
    .expect("chunk count")
}

#[tokio::test]
async fn appends_chunks_and_advances_the_source() {
  let (store, bank, source) = setup().await;
  let chunks = vec![
    chunk(bank.id, source.id, 0, "hash-1", 2_000),
    chunk(bank.id, source.id, 1, "hash-2", 3_000),
  ];

  let inserted = store
    .append_chunks(source.id, chunks)
    .await
    .expect("append chunks");

  assert_eq!(inserted, 2);
  assert_eq!(chunk_count(&store, bank.id).await, 2);
  assert_eq!(
    source_status(&store, source.id).await,
    ("chunked".to_string(), 3_000)
  );

  let pending: i64 =
    sqlx::query_scalar("SELECT COUNT(*) FROM chunk WHERE bank_id = ? AND status = 'pending'")
      .bind(bank.id.to_string())
      .fetch_one(store.pool())
      .await
      .expect("count pending");
  assert_eq!(pending, 2);
}

#[tokio::test]
async fn replaying_the_same_batch_inserts_nothing_but_still_advances() {
  let (store, bank, source) = setup().await;
  let first = vec![
    chunk(bank.id, source.id, 0, "hash-1", 2_000),
    chunk(bank.id, source.id, 1, "hash-2", 3_000),
  ];
  store
    .append_chunks(source.id, first)
    .await
    .expect("first append");

  let replay = vec![
    chunk(bank.id, source.id, 0, "hash-1", 9_000),
    chunk(bank.id, source.id, 1, "hash-2", 9_000),
  ];
  let inserted = store
    .append_chunks(source.id, replay)
    .await
    .expect("replay");

  assert_eq!(inserted, 0, "hash conflicts are skipped");
  assert_eq!(chunk_count(&store, bank.id).await, 2);
  assert_eq!(
    source_status(&store, source.id).await,
    ("chunked".to_string(), 9_000),
    "updated_at comes from the attempted batch"
  );
}

#[tokio::test]
async fn cross_source_identical_content_is_skipped() {
  let (store, bank, first) = setup().await;
  let second = source(bank.id, "same payload");
  store
    .create_source(second.clone())
    .await
    .expect("create second source");

  let inserted = store
    .append_chunks(first.id, vec![chunk(bank.id, first.id, 0, "shared", 2_000)])
    .await
    .expect("first source chunk");
  assert_eq!(inserted, 1);

  let inserted = store
    .append_chunks(
      second.id,
      vec![chunk(bank.id, second.id, 0, "shared", 4_000)],
    )
    .await
    .expect("second source chunk");

  assert_eq!(inserted, 0, "material already belongs to the first source");
  assert_eq!(chunk_count(&store, bank.id).await, 1);
  let owner: String = sqlx::query_scalar("SELECT source_id FROM chunk WHERE bank_id = ?")
    .bind(bank.id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("chunk owner");
  assert_eq!(
    owner,
    first.id.to_string(),
    "the surviving row belongs to the first source"
  );
  assert_eq!(
    source_status(&store, second.id).await,
    ("chunked".to_string(), 4_000),
    "a zero-chunk source is still a ledger row"
  );
}

#[tokio::test]
async fn partial_conflicts_insert_only_new_rows() {
  let (store, bank, source) = setup().await;
  store
    .append_chunks(
      source.id,
      vec![chunk(bank.id, source.id, 0, "hash-1", 2_000)],
    )
    .await
    .expect("seed chunk");

  let inserted = store
    .append_chunks(
      source.id,
      vec![
        chunk(bank.id, source.id, 0, "hash-1", 3_000),
        chunk(bank.id, source.id, 1, "hash-2", 4_000),
      ],
    )
    .await
    .expect("partial append");

  assert_eq!(inserted, 1);
  assert_eq!(chunk_count(&store, bank.id).await, 2);
  assert_eq!(
    source_status(&store, source.id).await,
    ("chunked".to_string(), 4_000)
  );
}

#[tokio::test]
async fn empty_batch_advances_status_but_keeps_updated_at() {
  let (store, bank, source) = setup().await;

  let inserted = store
    .append_chunks(source.id, Vec::new())
    .await
    .expect("empty append");

  assert_eq!(inserted, 0);
  assert_eq!(chunk_count(&store, bank.id).await, 0);
  assert_eq!(
    source_status(&store, source.id).await,
    ("chunked".to_string(), SOURCE_TIME)
  );
}

#[tokio::test]
async fn missing_source_is_not_found_and_writes_nothing() {
  let (store, bank, _source) = setup().await;
  let missing = SourceId::new();

  let error = store
    .append_chunks(missing, vec![chunk(bank.id, missing, 0, "hash-1", 2_000)])
    .await
    .expect_err("missing source");

  assert!(matches!(error, Error::NotFound("source", _)));
  assert_eq!(chunk_count(&store, bank.id).await, 0, "nothing was written");
}

#[tokio::test]
async fn chunks_from_another_bank_are_rejected() {
  let (store, bank, source) = setup().await;
  let mut chunk = chunk(bank.id, source.id, 0, "hash-1", 2_000);
  chunk.bank_id = BankId::new();

  let error = store
    .append_chunks(source.id, vec![chunk])
    .await
    .expect_err("mismatched bank");

  assert!(matches!(error, Error::Config(_)));
  assert_eq!(chunk_count(&store, bank.id).await, 0);
  assert_eq!(
    source_status(&store, source.id).await,
    ("received".to_string(), SOURCE_TIME),
    "nothing changed"
  );
}

#[tokio::test]
async fn chunk_fields_are_persisted_verbatim() {
  let (store, bank, source) = setup().await;
  let mut chunk = chunk(bank.id, source.id, 7, "hash-7", 5_000);
  chunk.text = "verbatim slice\nwith newline".into();
  chunk.time_start = Some(timestamp(4_000));
  chunk.time_end = Some(timestamp(5_000));

  store
    .append_chunks(source.id, vec![chunk.clone()])
    .await
    .expect("append chunk");

  type Row = (
    String,
    String,
    String,
    i64,
    String,
    String,
    Option<i64>,
    Option<i64>,
    i64,
    i64,
  );
  let row = sqlx::query_as::<_, Row>(
    "SELECT id, bank_id, source_id, chunk_index, text, status, time_start, time_end, \
     created_at, updated_at FROM chunk WHERE id = ?",
  )
  .bind(chunk.id.to_string())
  .fetch_one(store.pool())
  .await
  .expect("chunk row");

  assert_eq!(row.0, chunk.id.to_string());
  assert_eq!(row.1, bank.id.to_string());
  assert_eq!(row.2, source.id.to_string());
  assert_eq!(row.3, 7);
  assert_eq!(row.4, "verbatim slice\nwith newline");
  assert_eq!(row.5, "pending");
  assert_eq!(row.6, Some(4_000));
  assert_eq!(row.7, Some(5_000));
  assert_eq!(row.8, 5_000);
  assert_eq!(row.9, 5_000);
}

#[tokio::test]
async fn mid_batch_failure_rolls_back_the_whole_batch() {
  let (store, bank, source) = setup().await;
  let first = chunk(bank.id, source.id, 0, "hash-1", 2_000);
  let mut second = chunk(bank.id, source.id, 1, "hash-2", 3_000);
  // Same primary key, different hash: the first insert succeeds, the second
  // violates the chunk PK and must roll the whole transaction back.
  second.id = first.id;

  let error = store
    .append_chunks(source.id, vec![first, second])
    .await
    .expect_err("duplicate chunk id");

  assert!(matches!(error, Error::Conflict("chunk", _)));
  assert_eq!(chunk_count(&store, bank.id).await, 0);
  assert_eq!(
    source_status(&store, source.id).await,
    ("received".to_string(), SOURCE_TIME),
    "status flip was rolled back too"
  );
}

#[tokio::test]
async fn chunks_from_another_source_are_rejected() {
  let (store, bank, source) = setup().await;
  let other = SourceId::new();

  let error = store
    .append_chunks(source.id, vec![chunk(bank.id, other, 0, "hash-1", 2_000)])
    .await
    .expect_err("mismatched source");

  assert!(matches!(error, Error::Config(_)));
  assert_eq!(chunk_count(&store, bank.id).await, 0);
  assert_eq!(
    source_status(&store, source.id).await,
    ("received".to_string(), SOURCE_TIME),
    "nothing changed"
  );
}
