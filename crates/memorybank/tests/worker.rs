//! Worker and rescan pass.

use std::sync::Arc;
use std::time::Duration;

use engine::ports::Store;
use engine::retain::RawPayload;
use engine::testing::{FakeLlm, FixedClock, HashEmbedder, timestamp};
use engine::types::{Bank, BankId, BankProfile, Source, SourceId, SourceStatus};
use memorybank::{MemoryBank, PipelineConfig, RetainConfig, RetainState, WorkerConfig};
use serde_json::{Value, json};
use store::testing::memory_pool;
use store::{EMBEDDING_DIM, SqliteStore};

fn ts(millis: i64) -> engine::types::Timestamp {
  timestamp(millis)
}

async fn setup(now: i64) -> (SqliteStore, Arc<dyn Store>, BankId, Arc<FakeLlm>) {
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
  (sqlite, store, bank.id, Arc::new(FakeLlm::new()))
}

fn memory(store: Arc<dyn Store>, llm: Arc<FakeLlm>, now: i64, worker: WorkerConfig) -> MemoryBank {
  let llm: Arc<dyn engine::ports::Llm> = llm;
  MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(now))),
    RetainConfig::default(),
  )
  .with_models(llm, Arc::new(HashEmbedder::new(EMBEDDING_DIM)))
  .with_pipeline(PipelineConfig::default())
  .with_worker_config(worker)
}

fn fact_json(what: &str, entity: &str) -> Value {
  json!({
    "what": what,
    "when": "during the conversation",
    "where": "Berlin",
    "who": entity,
    "why": "it matters",
    "fact_type": "world",
    "entities": [{ "text": entity, "kind": "person" }]
  })
}

fn payload(text: &str) -> RawPayload {
  RawPayload::Document {
    title: Some("note".into()),
    text: text.into(),
  }
}

#[tokio::test]
async fn retain_then_rescan_extracts_and_commits() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  llm.push(
    "fact_extraction",
    json!({ "facts": [fact_json("Emily joined Google", "Emily"), fact_json("Emily likes Rust", "Emily")] }),
  );
  let memory = memory(
    store,
    llm.clone(),
    1_000,
    WorkerConfig {
      max_attempts: 1,
      ..WorkerConfig::default()
    },
  );

  let source = memory
    .retain(bank, payload("Emily joined Google and likes Rust."))
    .await
    .expect("retain");
  let report = memory.rescan(bank).await.expect("rescan");

  assert_eq!(report.sources_chunked, 0);
  assert_eq!(report.chunks_processed, 1);
  assert_eq!(report.failed, 0);

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(sqlite.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 2);
  let vectors: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_vec")
    .fetch_one(sqlite.pool())
    .await
    .expect("vectors");
  assert_eq!(vectors, 2);
  let entities: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entity")
    .fetch_one(sqlite.pool())
    .await
    .expect("entities");
  assert_eq!(entities, 1, "both facts mention the same entity");
  let mention_count: i64 = sqlx::query_scalar("SELECT mention_count FROM entity")
    .fetch_one(sqlite.pool())
    .await
    .expect("stats");
  assert_eq!(mention_count, 2);
  let bindings: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity")
    .fetch_one(sqlite.pool())
    .await
    .expect("bindings");
  assert_eq!(bindings, 2);
  let fts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_fts")
    .fetch_one(sqlite.pool())
    .await
    .expect("fts");
  assert_eq!(fts, 2);

  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::Completed);
  assert_eq!(status.facts_written, 2);
}

#[tokio::test]
async fn a_failed_extraction_is_dead_lettered_and_not_retried() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  // No scripted extraction response: FakeLlm exhausts the queue.
  let memory = memory(
    store,
    llm.clone(),
    1_000,
    WorkerConfig {
      max_attempts: 1,
      ..WorkerConfig::default()
    },
  );

  let source = memory
    .retain(bank, payload("will fail"))
    .await
    .expect("retain");
  let report = memory.rescan(bank).await.expect("rescan");
  assert_eq!(report.failed, 1);
  assert_eq!(report.chunks_processed, 0);

  let status: String = sqlx::query_scalar("SELECT status FROM chunk LIMIT 1")
    .fetch_one(sqlite.pool())
    .await
    .expect("chunk");
  assert_eq!(status, "failed");
  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(sqlite.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 0);

  let second = memory.rescan(bank).await.expect("second rescan");
  assert_eq!(second.chunks_processed, 0);
  assert_eq!(second.failed, 0, "failed chunks are not retried");
  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::Failed);
  let _ = llm;
}

#[tokio::test]
async fn stale_extraction_is_reclaimed() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  llm.push(
    "fact_extraction",
    json!({ "facts": [fact_json("recovered", "Emily")] }),
  );
  let memory = memory(
    store.clone(),
    llm.clone(),
    1_000,
    WorkerConfig {
      stale_after: Duration::from_millis(500),
      max_attempts: 1,
      ..WorkerConfig::default()
    },
  );

  memory
    .retain(bank, payload("stale work"))
    .await
    .expect("retain");
  let claimed = sqlite
    .claim_chunk(bank, ts(1_000), Duration::from_secs(3_600))
    .await
    .expect("claim")
    .expect("chunk");
  sqlx::query("UPDATE chunk SET updated_at = 0 WHERE id = ?")
    .bind(claimed.chunk.id.to_string())
    .execute(sqlite.pool())
    .await
    .expect("age the claim");

  let report = memory.rescan(bank).await.expect("rescan");
  assert_eq!(report.chunks_processed, 1);
  assert_eq!(report.failed, 0);

  let status: String = sqlx::query_scalar("SELECT status FROM chunk WHERE id = ?")
    .bind(claimed.chunk.id.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunk");
  assert_eq!(status, "done");
}

#[tokio::test]
async fn received_sources_are_rechunked_by_rescan() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  llm.push(
    "fact_extraction",
    json!({ "facts": [fact_json("from a crashed tx1", "Emily")] }),
  );
  let memory = memory(
    store.clone(),
    llm.clone(),
    1_000,
    WorkerConfig {
      max_attempts: 1,
      ..WorkerConfig::default()
    },
  );

  let source = SourceId::new();
  store
    .create_source(Source {
      id: source,
      bank_id: bank,
      title: None,
      payload_text: "recovered payload".into(),
      status: SourceStatus::Received,
      created_at: ts(0),
      updated_at: ts(0),
    })
    .await
    .expect("source");

  let report = memory.rescan(bank).await.expect("rescan");
  assert_eq!(report.sources_chunked, 1);
  assert_eq!(report.chunks_processed, 1);

  let status: String = sqlx::query_scalar("SELECT status FROM source WHERE id = ?")
    .bind(source.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("source row");
  assert_eq!(status, "chunked");
  let chunks: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chunk WHERE source_id = ?")
    .bind(source.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunks");
  assert_eq!(chunks, 1);
}

#[tokio::test]
async fn worker_drains_a_queued_source() {
  let (_, store, bank, llm) = setup(1_000).await;
  llm.push(
    "fact_extraction",
    json!({ "facts": [fact_json("worker ran", "Emily")] }),
  );
  let memory = memory(
    store.clone(),
    llm.clone(),
    1_000,
    WorkerConfig {
      max_attempts: 1,
      poll_interval: Duration::from_millis(10),
      retry_delay: Duration::from_millis(10),
      ..WorkerConfig::default()
    },
  );

  let handle = memory.spawn_worker(WorkerConfig {
    max_attempts: 1,
    poll_interval: Duration::from_millis(10),
    retry_delay: Duration::from_millis(10),
    ..WorkerConfig::default()
  });

  let source = memory
    .retain(bank, payload("worker payload"))
    .await
    .expect("retain");

  tokio::time::timeout(Duration::from_secs(3), async {
    loop {
      let status = memory.retain_status(bank, source).await.expect("status");
      if status.state == RetainState::Completed {
        break;
      }
      tokio::time::sleep(Duration::from_millis(10)).await;
    }
  })
  .await
  .expect("worker converged");

  handle.abort();
}

#[tokio::test]
async fn rescan_requires_models() {
  let (sqlite, store, bank, _llm) = setup(1_000).await;
  let memory = MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(1_000))),
    RetainConfig::default(),
  );

  let source = memory
    .retain(bank, payload("no models"))
    .await
    .expect("retain");
  let report = memory.rescan(bank).await.expect("rescan");
  assert_eq!(report.failed, 1);

  let status: String = sqlx::query_scalar("SELECT status FROM chunk WHERE source_id = ?")
    .bind(source.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunk");
  assert_eq!(status, "failed");
}
