//! Facade retain ack and status derivation.

use std::sync::Arc;
use std::time::Duration;

use engine::ports::Store;
use engine::retain::{RawPayload, RawTurn};
use engine::testing::{FixedClock, timestamp};
use engine::types::{Bank, BankId, BankProfile, ChunkId, FactId, Source, SourceId, SourceStatus};
use memorybank::{MemoryBank, RetainConfig, RetainState};
use store::SqliteStore;
use store::testing::memory_pool;

fn ts(millis: i64) -> engine::types::Timestamp {
  timestamp(millis)
}

async fn setup(now: i64) -> (SqliteStore, Arc<dyn Store>, BankId) {
  let sqlite = SqliteStore::new(memory_pool().await.expect("pool"));
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
    created_at: ts(now),
    updated_at: ts(now),
  };
  sqlite.create_bank(bank.clone()).await.expect("create bank");
  let store: Arc<dyn Store> = Arc::new(sqlite.clone());
  (sqlite, store, bank.id)
}

fn memory_bank(store: Arc<dyn Store>, now: i64) -> MemoryBank {
  MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(now))),
    RetainConfig::default(),
  )
}

fn payload(text: &str) -> RawPayload {
  RawPayload::Transcript {
    title: Some("transcript".into()),
    turns: vec![RawTurn {
      speaker: "user".into(),
      text: text.into(),
      at: Some(ts(1_000)),
      session: Some(1),
    }],
  }
}

#[tokio::test]
async fn retain_acks_after_chunks_are_persisted() {
  let (sqlite, store, bank) = setup(1_000).await;
  let memory = memory_bank(store, 1_000);

  let source = memory
    .retain(bank, payload("hello world"))
    .await
    .expect("retain");

  let (status, text): (String, String) =
    sqlx::query_as("SELECT status, payload_text FROM source WHERE id = ?")
      .bind(source.to_string())
      .fetch_one(sqlite.pool())
      .await
      .expect("source row");
  assert_eq!(status, "chunked");
  assert!(text.contains("hello world"));

  let chunks: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunk WHERE source_id = ?")
    .bind(source.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunks");
  assert!(chunks > 0);
  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(sqlite.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 0, "the ack does not run extraction");

  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::Queued);
  assert_eq!(status.chunks_total as i64, chunks);
  assert_eq!(status.chunks_pending as i64, chunks);
  assert_eq!(status.chunks_extracting, 0);
  assert_eq!(status.chunks_done, 0);
  assert_eq!(status.chunks_failed, 0);
  assert_eq!(status.facts_written, 0);
}

#[tokio::test]
async fn status_tracks_claims_staleness_and_completion() {
  let (sqlite, store, bank) = setup(1_000).await;
  let memory = memory_bank(store.clone(), 1_000);
  let source = memory
    .retain(bank, payload("status test"))
    .await
    .expect("retain");
  let chunk: String = sqlx::query_scalar("SELECT id FROM chunk WHERE source_id = ? LIMIT 1")
    .bind(source.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunk id");

  let claimed = sqlite
    .claim_chunk(bank, ts(1_000), Duration::from_secs(3_600))
    .await
    .expect("claim")
    .expect("pending chunk");
  assert_eq!(claimed.chunk.id, ChunkId::parse(&chunk).expect("id"));

  let status = memory.retain_status(bank, source).await.expect("running");
  assert_eq!(status.state, RetainState::Running);
  assert_eq!(status.chunks_extracting, 1);

  // Default stale_after is 10 minutes; step past it.
  let later = memory_bank(store, 1_000 + 601_000);
  let status = later.retain_status(bank, source).await.expect("stalled");
  assert_eq!(status.state, RetainState::Stalled);

  sqlx::query("UPDATE chunk SET status = 'done', updated_at = 2_000 WHERE id = ?")
    .bind(&chunk)
    .execute(sqlite.pool())
    .await
    .expect("mark done");
  let status = memory.retain_status(bank, source).await.expect("completed");
  assert_eq!(status.state, RetainState::Completed);
  assert_eq!(status.chunks_done, 1);
}

#[tokio::test]
async fn re_ingesting_identical_material_is_completed_with_zero_chunks() {
  let (sqlite, store, bank) = setup(1_000).await;
  let memory = memory_bank(store, 1_000);

  memory
    .retain(bank, payload("same material"))
    .await
    .expect("first");
  let second = memory
    .retain(bank, payload("same material"))
    .await
    .expect("second");

  let status = memory.retain_status(bank, second).await.expect("status");
  assert_eq!(status.chunks_total, 0);
  assert_eq!(status.state, RetainState::Completed);

  let chunks: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunk WHERE source_id = ?")
    .bind(second.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunks");
  assert_eq!(chunks, 0, "all hashes collided with the first source");
}

#[tokio::test]
async fn unknown_source_is_not_found() {
  let (_, store, bank) = setup(1_000).await;
  let memory = memory_bank(store, 1_000);

  let error = memory
    .retain_status(bank, SourceId::new())
    .await
    .expect_err("unknown source");

  assert!(
    matches!(error, engine::Error::NotFound(_, _)),
    "error: {error}"
  );
}

#[tokio::test]
async fn a_received_source_reports_received() {
  let (sqlite, store, bank) = setup(1_000).await;
  let memory = memory_bank(store, 1_000);
  let source = SourceId::new();
  sqlite
    .create_source(Source {
      id: source,
      bank_id: bank,
      title: None,
      payload_text: "not chunked yet".into(),
      status: SourceStatus::Received,
      created_at: ts(0),
      updated_at: ts(0),
    })
    .await
    .expect("source");

  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::Received);
  assert_eq!(status.chunks_total, 0);
}

#[tokio::test]
async fn mixed_done_and_failed_chunks_report_the_right_state() {
  let (sqlite, store, bank) = setup(1_000).await;
  let memory = memory_bank(store, 1_000);
  let source = memory
    .retain(bank, payload("mixed result"))
    .await
    .expect("retain");
  let first: String = sqlx::query_scalar("SELECT id FROM chunk WHERE source_id = ? LIMIT 1")
    .bind(source.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunk id");

  sqlx::query(
    "INSERT INTO chunk (id, bank_id, source_id, chunk_index, text, content_hash, status, \
     created_at, updated_at) VALUES (?, ?, ?, 1, 'extra', 'extra-hash', 'pending', 0, 0)",
  )
  .bind(ChunkId::new().to_string())
  .bind(bank.to_string())
  .bind(source.to_string())
  .execute(sqlite.pool())
  .await
  .expect("second chunk");

  sqlx::query("UPDATE chunk SET status = 'done' WHERE id = ?")
    .bind(&first)
    .execute(sqlite.pool())
    .await
    .expect("done");
  sqlx::query("UPDATE chunk SET status = 'failed' WHERE source_id = ? AND id <> ?")
    .bind(source.to_string())
    .bind(&first)
    .execute(sqlite.pool())
    .await
    .expect("failed");

  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::CompletedWithFailures);
  assert_eq!(status.chunks_done, 1);
  assert_eq!(status.chunks_failed, 1);

  sqlx::query("UPDATE chunk SET status = 'failed' WHERE source_id = ?")
    .bind(source.to_string())
    .execute(sqlite.pool())
    .await
    .expect("all failed");
  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::Failed);
  assert_eq!(status.chunks_done, 0);
}

#[tokio::test]
async fn facts_written_counts_committed_facts() {
  let (sqlite, store, bank) = setup(1_000).await;
  let memory = memory_bank(store, 1_000);
  let source = memory
    .retain(bank, payload("fact count"))
    .await
    .expect("retain");
  let chunk: String = sqlx::query_scalar("SELECT id FROM chunk WHERE source_id = ? LIMIT 1")
    .bind(source.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunk id");

  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, chunk_id, created_at, updated_at) \
     VALUES (?, ?, 'world', 'a fact', ?, 0, 0)",
  )
  .bind(FactId::new().to_string())
  .bind(bank.to_string())
  .bind(&chunk)
  .execute(sqlite.pool())
  .await
  .expect("fact");

  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.facts_written, 1);
}
