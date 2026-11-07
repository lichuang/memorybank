//! `release_chunk` / `fail_chunk`: claim-token guarded transitions.

use std::time::Duration;

use engine::Error;
use engine::ports::{NewChunk, Store};
use engine::testing::timestamp;
use engine::types::{
  Bank, BankId, BankProfile, Chunk, ChunkId, ChunkStatus, Source, SourceId, SourceStatus,
};
use store::SqliteStore;
use store::testing::memory_pool;

async fn setup() -> (SqliteStore, BankId, SourceId) {
  let store = SqliteStore::new(memory_pool().await.expect("in-memory store"));
  let bank = Bank {
    id: BankId::new(),
    name: "primary".into(),
    profile: BankProfile {
      background: "background".into(),
      skepticism: 3,
      literalism: 3,
      empathy: 3,
      bias_strength: 0.5,
    },
    created_at: timestamp(0),
    updated_at: timestamp(0),
  };
  store.create_bank(bank.clone()).await.expect("create bank");

  let source = Source {
    id: SourceId::new(),
    bank_id: bank.id,
    title: None,
    payload_text: "payload".into(),
    status: SourceStatus::Received,
    created_at: timestamp(0),
    updated_at: timestamp(0),
  };
  store
    .create_source(source.clone())
    .await
    .expect("create source");

  (store, bank.id, source.id)
}

async fn seed_chunk(store: &SqliteStore, bank: BankId, source: SourceId) -> ChunkId {
  let chunk = NewChunk {
    id: ChunkId::new(),
    bank_id: bank,
    source_id: source,
    chunk_index: 0,
    text: "text".into(),
    content_hash: "hash-1".into(),
    time_start: None,
    time_end: None,
    created_at: timestamp(1_000),
  };
  let id = chunk.id;
  store
    .append_chunks(source, vec![chunk])
    .await
    .expect("seed chunk");
  id
}

async fn chunk_row(store: &SqliteStore, id: ChunkId) -> (String, i64) {
  sqlx::query_as("SELECT status, updated_at FROM chunk WHERE id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("chunk row")
}

async fn claim(store: &SqliteStore, bank: BankId) -> Chunk {
  store
    .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
    .await
    .expect("claim")
    .expect("chunk")
    .chunk
}

#[tokio::test]
async fn releases_a_claimed_chunk_back_to_pending() {
  let (store, bank, source) = setup().await;
  let id = seed_chunk(&store, bank, source).await;
  let claimed = claim(&store, bank).await;

  store
    .release_chunk(id, claimed.updated_at, timestamp(6_000))
    .await
    .expect("release");

  assert_eq!(
    chunk_row(&store, id).await,
    ("pending".to_string(), 6_000),
    "released and timestamp refreshed"
  );

  let reclaimed = store
    .claim_chunk(bank, timestamp(7_000), Duration::ZERO)
    .await
    .expect("claim")
    .expect("chunk is claimable again");
  assert_eq!(reclaimed.chunk.id, id);
  assert_eq!(reclaimed.chunk.status, ChunkStatus::Extracting);
  assert_eq!(reclaimed.chunk.updated_at, timestamp(7_000));
}

#[tokio::test]
async fn repeated_release_is_idempotent() {
  let (store, bank, source) = setup().await;
  let id = seed_chunk(&store, bank, source).await;
  let claimed = claim(&store, bank).await;

  store
    .release_chunk(id, claimed.updated_at, timestamp(6_000))
    .await
    .expect("first release");
  store
    .release_chunk(id, claimed.updated_at, timestamp(9_000))
    .await
    .expect("second release is a no-op");

  assert_eq!(
    chunk_row(&store, id).await,
    ("pending".to_string(), 6_000),
    "the no-op does not refresh updated_at"
  );
}

#[tokio::test]
async fn a_stale_token_cannot_release() {
  let (store, bank, source) = setup().await;
  let id = seed_chunk(&store, bank, source).await;
  let claimed = claim(&store, bank).await;
  let stale = timestamp(claimed.updated_at.timestamp_millis() + 1);

  let error = store
    .release_chunk(id, stale, timestamp(6_000))
    .await
    .expect_err("stale token");

  assert!(matches!(error, Error::Conflict(_, _)), "error: {error}");
  assert_eq!(chunk_row(&store, id).await.0, "extracting");

  store
    .release_chunk(id, claimed.updated_at, timestamp(6_000))
    .await
    .expect("the real owner can release");
  assert_eq!(chunk_row(&store, id).await.0, "pending");
}

#[tokio::test]
async fn done_chunks_are_not_resurrected() {
  let (store, bank, source) = setup().await;
  let id = seed_chunk(&store, bank, source).await;
  sqlx::query("UPDATE chunk SET status = 'done' WHERE id = ?")
    .bind(id.to_string())
    .execute(store.pool())
    .await
    .expect("mark done");

  store
    .release_chunk(id, timestamp(0), timestamp(6_000))
    .await
    .expect("no-op");

  assert_eq!(chunk_row(&store, id).await.0, "done");
}

#[tokio::test]
async fn failed_chunks_are_not_resurrected() {
  let (store, bank, source) = setup().await;
  let id = seed_chunk(&store, bank, source).await;
  sqlx::query("UPDATE chunk SET status = 'failed' WHERE id = ?")
    .bind(id.to_string())
    .execute(store.pool())
    .await
    .expect("mark failed");

  store
    .release_chunk(id, timestamp(0), timestamp(6_000))
    .await
    .expect("no-op");

  assert_eq!(chunk_row(&store, id).await.0, "failed");
}

#[tokio::test]
async fn missing_chunk_is_not_found() {
  let (store, _, _) = setup().await;

  let error = store
    .release_chunk(ChunkId::new(), timestamp(0), timestamp(6_000))
    .await
    .expect_err("missing chunk");

  assert!(matches!(error, Error::NotFound("chunk", _)));
}

#[tokio::test]
async fn fail_chunk_dead_letters_an_extracting_chunk() {
  let (store, bank, source) = setup().await;
  let id = seed_chunk(&store, bank, source).await;
  let claimed = claim(&store, bank).await;

  store
    .fail_chunk(id, claimed.updated_at, timestamp(6_000))
    .await
    .expect("fail");

  assert_eq!(chunk_row(&store, id).await, ("failed".to_string(), 6_000));
}

#[tokio::test]
async fn fail_chunk_rejects_a_stale_token() {
  let (store, bank, source) = setup().await;
  let id = seed_chunk(&store, bank, source).await;
  let claimed = claim(&store, bank).await;
  let stale = timestamp(claimed.updated_at.timestamp_millis() + 1);

  let error = store
    .fail_chunk(id, stale, timestamp(6_000))
    .await
    .expect_err("stale token");

  assert!(matches!(error, Error::Conflict(_, _)), "error: {error}");
  assert_eq!(chunk_row(&store, id).await.0, "extracting");
}

#[tokio::test]
async fn fail_chunk_rejects_a_non_extracting_chunk() {
  let (store, bank, source) = setup().await;
  let id = seed_chunk(&store, bank, source).await;
  sqlx::query("UPDATE chunk SET status = 'done' WHERE id = ?")
    .bind(id.to_string())
    .execute(store.pool())
    .await
    .expect("mark done");

  let error = store
    .fail_chunk(id, timestamp(0), timestamp(6_000))
    .await
    .expect_err("not extracting");

  assert!(matches!(error, Error::Conflict(_, _)), "error: {error}");
}

#[tokio::test]
async fn fail_chunk_missing_is_not_found() {
  let (store, _, _) = setup().await;

  let error = store
    .fail_chunk(ChunkId::new(), timestamp(0), timestamp(6_000))
    .await
    .expect_err("missing chunk");

  assert!(matches!(error, Error::NotFound("chunk", _)));
}
