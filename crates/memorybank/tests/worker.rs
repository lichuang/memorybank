//! Worker and rescan pass.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use engine::link::LinkConfig;
use engine::ports::{JsonRequest, Llm, Store};
use engine::retain::{BackgroundConfig, ChunkConfig, RawPayload, RawTurn, normalize};
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
  memory_with_pipeline(store, llm, now, worker, PipelineConfig::default())
}

fn memory_with_pipeline(
  store: Arc<dyn Store>,
  llm: Arc<FakeLlm>,
  now: i64,
  worker: WorkerConfig,
  pipeline: PipelineConfig,
) -> MemoryBank {
  let llm: Arc<dyn engine::ports::Llm> = llm;
  MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(now))),
    RetainConfig::default(),
  )
  .with_models(llm, Arc::new(HashEmbedder::new(EMBEDDING_DIM)))
  .with_pipeline(pipeline)
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

fn fact_json_at(what: &str, entity: &str, at: engine::types::Timestamp) -> Value {
  let mut value = fact_json(what, entity);
  value["occurred_start"] = json!(at.to_rfc3339());
  value["occurred_end"] = json!(at.to_rfc3339());
  value
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
async fn end_to_end_commits_temporal_links_and_consistent_entity_stats() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  let first_at = ts(1_000);
  let second_at = ts(1_000 + 3_600_000);
  llm.push(
    "fact_extraction",
    json!({
      "facts": [
        fact_json_at("Alice met Bob", "Alice", first_at),
        fact_json_at("Alice called Bob", "Alice", second_at)
      ]
    }),
  );

  let pipeline = PipelineConfig {
    links: LinkConfig {
      semantic_top_k: 0,
      ..LinkConfig::default()
    },
    ..PipelineConfig::default()
  };
  let memory = memory_with_pipeline(
    store,
    llm.clone(),
    1_000,
    WorkerConfig {
      max_attempts: 1,
      ..WorkerConfig::default()
    },
    pipeline,
  );

  let source = memory
    .retain(
      bank,
      payload("Alice met Bob, then called Bob an hour later."),
    )
    .await
    .expect("retain");
  let report = memory.rescan(bank).await.expect("rescan");
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
  let fts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_fts")
    .fetch_one(sqlite.pool())
    .await
    .expect("fts");
  assert_eq!(fts, 2);

  let entities: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entity")
    .fetch_one(sqlite.pool())
    .await
    .expect("entities");
  assert_eq!(entities, 1);
  let bindings: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity")
    .fetch_one(sqlite.pool())
    .await
    .expect("bindings");
  assert_eq!(bindings, 2);

  let (mention_count, bound): (i64, i64) = sqlx::query_as(
    "SELECT e.mention_count, \
       (SELECT COUNT(*) FROM fact_entity fe WHERE fe.entity_id = e.id) \
     FROM entity e",
  )
  .fetch_one(sqlite.pool())
  .await
  .expect("entity stats");
  assert_eq!(
    mention_count, bound,
    "mention_count stays consistent with bindings"
  );
  assert_eq!(mention_count, 2);

  let temporal: i64 =
    sqlx::query_scalar("SELECT COUNT(*) FROM fact_link WHERE link_type = 'temporal'")
      .fetch_one(sqlite.pool())
      .await
      .expect("temporal links");
  assert_eq!(temporal, 2, "one temporal pair is stored as two rows");

  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::Completed);
  assert_eq!(status.facts_written, 2);
}

#[tokio::test]
async fn incremental_reingest_processes_only_new_chunks() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  let memory = memory(
    store,
    llm.clone(),
    1_000,
    WorkerConfig {
      max_attempts: 1,
      ..WorkerConfig::default()
    },
  );

  llm.push(
    "fact_extraction",
    json!({ "facts": [fact_json("first source fact", "Alice")] }),
  );
  let first = memory
    .retain(bank, payload("first material"))
    .await
    .expect("first retain");
  let first_report = memory.rescan(bank).await.expect("first rescan");
  assert_eq!(first_report.chunks_processed, 1);

  llm.push(
    "fact_extraction",
    json!({ "facts": [fact_json("second source fact", "Alice")] }),
  );
  let second = memory
    .retain(bank, payload("second material"))
    .await
    .expect("second retain");
  let second_report = memory.rescan(bank).await.expect("second rescan");
  assert_eq!(
    second_report.chunks_processed, 1,
    "only the new source's chunk is processed"
  );

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(sqlite.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 2);
  let first_status = memory.retain_status(bank, first).await.expect("first");
  assert_eq!(first_status.state, RetainState::Completed);
  let second_status = memory.retain_status(bank, second).await.expect("second");
  assert_eq!(second_status.state, RetainState::Completed);

  let duplicate = memory
    .retain(bank, payload("first material"))
    .await
    .expect("duplicate retain");
  let duplicate_status = memory
    .retain_status(bank, duplicate)
    .await
    .expect("duplicate status");
  assert_eq!(duplicate_status.chunks_total, 0);
  assert_eq!(duplicate_status.state, RetainState::Completed);
  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(sqlite.pool())
    .await
    .expect("facts after duplicate");
  assert_eq!(facts, 2, "re-ingest does not duplicate facts");

  let extraction_calls = llm
    .calls()
    .iter()
    .filter(|call| call.schema_name == "fact_extraction")
    .count();
  assert_eq!(
    extraction_calls, 2,
    "the zero-chunk source makes no LLM call"
  );
}

#[tokio::test]
async fn mixed_chunk_outcomes_report_completed_with_failures() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  llm.push(
    "fact_extraction",
    json!({ "facts": [fact_json("surviving fact", "Alice")] }),
  );

  let retain = RetainConfig {
    chunk: ChunkConfig {
      max_tokens: 50,
      min_tokens: 0,
      ..ChunkConfig::default()
    },
    ..RetainConfig::default()
  };
  let llm_port: Arc<dyn Llm> = llm.clone();
  let memory = MemoryBank::new(store, Arc::new(FixedClock(ts(1_000))), retain)
    .with_models(llm_port, Arc::new(HashEmbedder::new(EMBEDDING_DIM)))
    .with_worker_config(WorkerConfig {
      max_attempts: 1,
      ..WorkerConfig::default()
    });

  let source = memory
    .retain(bank, payload(&"alpha beta gamma delta ".repeat(20)))
    .await
    .expect("retain");
  let report = memory.rescan(bank).await.expect("rescan");
  assert_eq!(report.chunks_processed, 1);
  assert!(report.failed >= 1, "the remaining chunks are dead-lettered");

  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::CompletedWithFailures);
  assert_eq!(status.chunks_done, 1);
  assert!(status.chunks_failed >= 1);
  assert!(
    status.chunks_total > 1,
    "the source produced multiple chunks"
  );

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(sqlite.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 1);
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

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(sqlite.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 1, "the stale claim is committed exactly once");
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
async fn received_transcript_recovers_chunk_times_on_rescan() {
  let (sqlite, store, bank, llm) = setup(500).await;
  llm.push(
    "fact_extraction",
    json!({ "facts": [fact_json("recovered from a crashed tx1", "Emily")] }),
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

  let payload = normalize(RawPayload::Transcript {
    title: None,
    turns: vec![
      RawTurn {
        speaker: "user".into(),
        text: "first turn".into(),
        at: Some(ts(1_000)),
        session: Some(1),
      },
      RawTurn {
        speaker: "assistant".into(),
        text: "second turn".into(),
        at: Some(ts(2_000)),
        session: Some(1),
      },
    ],
  });
  let source = SourceId::new();
  sqlite
    .create_source(Source {
      id: source,
      bank_id: bank,
      title: None,
      payload_text: payload.text,
      status: SourceStatus::Received,
      created_at: ts(0),
      updated_at: ts(0),
    })
    .await
    .expect("source");

  let report = memory.rescan(bank).await.expect("rescan");
  assert_eq!(report.sources_chunked, 1);
  assert_eq!(report.chunks_processed, 1);

  let (start, end): (Option<i64>, Option<i64>) =
    sqlx::query_as("SELECT time_start, time_end FROM chunk WHERE source_id = ?")
      .bind(source.to_string())
      .fetch_one(sqlite.pool())
      .await
      .expect("chunk times");
  assert_eq!(start, Some(1_000), "the first timestamp is recovered");
  assert_eq!(end, Some(2_000), "the last timestamp is recovered");

  let mentioned: Option<i64> = sqlx::query_scalar(
    "SELECT f.mentioned_at FROM fact f JOIN chunk c ON c.id = f.chunk_id \
     WHERE c.source_id = ?",
  )
  .bind(source.to_string())
  .fetch_one(sqlite.pool())
  .await
  .expect("fact mention time");
  assert_eq!(
    mentioned,
    Some(1_500),
    "normalize_time fills the midpoint from recovered chunk bounds"
  );
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

#[tokio::test]
async fn opinion_confidence_is_committed() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  llm.push(
    "fact_extraction",
    json!({ "facts": [{
      "what": "Rust is a good language",
      "when": "during the conversation",
      "where": "Berlin",
      "who": "the user",
      "why": "they enjoy working with it",
      "fact_type": "opinion",
      "confidence": 0.7
    }] }),
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

  memory
    .retain(bank, payload("I like Rust"))
    .await
    .expect("retain");
  memory.rescan(bank).await.expect("rescan");

  let confidence: f64 =
    sqlx::query_scalar("SELECT confidence FROM fact WHERE fact_type = 'opinion'")
      .fetch_one(sqlite.pool())
      .await
      .expect("opinion row");
  assert!((confidence - 0.7).abs() < 1e-6);
}

#[tokio::test]
async fn background_snippets_update_the_bank_profile_with_the_chunk() {
  let (sqlite, store, bank, llm) = setup(500).await;
  llm.push(
    "fact_extraction",
    json!({
      "facts": [],
      "background_snippets": [
        "I was born in Texas.",
        "I have ten years of startup experience."
      ]
    }),
  );
  llm.push(
    "background_merge",
    json!({ "background": "I was born in Texas and have ten years of startup experience." }),
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
    .retain(
      bank,
      payload("I was born in Texas and have ten years of startup experience."),
    )
    .await
    .expect("retain");
  let report = memory.rescan(bank).await.expect("rescan");

  assert_eq!(report.chunks_processed, 1);
  assert_eq!(report.failed, 0);

  let (background, updated_at): (String, i64) =
    sqlx::query_as("SELECT background, updated_at FROM bank WHERE id = ?")
      .bind(bank.to_string())
      .fetch_one(sqlite.pool())
      .await
      .expect("bank");
  assert_eq!(
    background,
    "I was born in Texas and have ten years of startup experience."
  );
  assert_eq!(updated_at, 1_000);

  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::Completed);
  assert_eq!(status.facts_written, 0);

  let calls = llm.calls();
  let merge = calls
    .iter()
    .find(|call| call.schema_name == "background_merge")
    .expect("background merge call");
  assert!(merge.user.contains("I was born in Texas."));
  assert!(
    merge
      .user
      .contains("I have ten years of startup experience.")
  );
}

#[tokio::test]
async fn no_background_snippets_leave_the_profile_untouched() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  llm.push(
    "fact_extraction",
    json!({ "facts": [fact_json("Emily joined Google", "Emily")] }),
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

  memory
    .retain(bank, payload("Emily joined Google."))
    .await
    .expect("retain");
  memory.rescan(bank).await.expect("rescan");

  let background: String = sqlx::query_scalar("SELECT background FROM bank WHERE id = ?")
    .bind(bank.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("bank");
  assert_eq!(background, "background");
  assert!(
    llm
      .calls()
      .iter()
      .all(|call| call.schema_name != "background_merge"),
    "no snippets means no merge call"
  );
}

#[tokio::test]
async fn a_failed_background_merge_does_not_commit_the_chunk() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  llm.push(
    "fact_extraction",
    json!({
      "facts": [fact_json("Emily joined Google", "Emily")],
      "background_snippets": ["I am an engineer."]
    }),
  );
  // No background_merge script: FakeLlm reports a deterministic failure.
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
    .retain(bank, payload("Emily joined Google. I am an engineer."))
    .await
    .expect("retain");
  let report = memory.rescan(bank).await.expect("rescan");
  assert_eq!(report.failed, 1);
  assert_eq!(report.chunks_processed, 0);

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(sqlite.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 0, "the merge failed before tx2");

  let entities: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entity")
    .fetch_one(sqlite.pool())
    .await
    .expect("entities");
  assert_eq!(entities, 0, "resolution proposals stay in memory until tx2");

  let background: String = sqlx::query_scalar("SELECT background FROM bank WHERE id = ?")
    .bind(bank.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("bank");
  assert_eq!(background, "background");

  let status: String = sqlx::query_scalar("SELECT status FROM chunk WHERE source_id = ?")
    .bind(source.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunk");
  assert_eq!(status, "failed");
}

/// Delegates to `FakeLlm` and simulates a concurrent background update on every
/// merge, so the tx2 CAS can never succeed.
struct AlwaysRacingLlm {
  inner: Arc<FakeLlm>,
  pool: sqlx::SqlitePool,
  bank: BankId,
}

#[async_trait]
impl Llm for AlwaysRacingLlm {
  async fn complete_json_raw(&self, req: JsonRequest) -> engine::Result<Value> {
    let response = self.inner.complete_json_raw(req.clone()).await?;
    if req.schema_name == "background_merge" {
      sqlx::query("UPDATE bank SET background = ?, updated_at = ? WHERE id = ?")
        .bind("I am from a racing update.")
        .bind(1_500_i64)
        .bind(self.bank.to_string())
        .execute(&self.pool)
        .await
        .expect("racing background update");
    }
    Ok(response)
  }

  fn model(&self) -> &str {
    "always-racing-llm"
  }
}

/// Delegates to `FakeLlm`, but simulates a concurrent chunk merging once while
/// the first background-merge call is in flight.
struct RacingLlm {
  inner: Arc<FakeLlm>,
  pool: sqlx::SqlitePool,
  bank: BankId,
  raced: AtomicBool,
}

#[async_trait]
impl Llm for RacingLlm {
  async fn complete_json_raw(&self, req: JsonRequest) -> engine::Result<Value> {
    let response = self.inner.complete_json_raw(req.clone()).await?;
    if req.schema_name == "background_merge" && !self.raced.swap(true, Ordering::SeqCst) {
      sqlx::query("UPDATE bank SET background = ?, updated_at = ? WHERE id = ?")
        .bind("I am from a concurrent chunk.")
        .bind(500_i64)
        .bind(self.bank.to_string())
        .execute(&self.pool)
        .await
        .expect("concurrent background update");
    }
    Ok(response)
  }

  fn model(&self) -> &str {
    "racing-llm"
  }
}

#[tokio::test]
async fn a_background_cas_conflict_is_remerged_and_retried() {
  let (sqlite, store, bank, llm) = setup(500).await;
  llm.push(
    "fact_extraction",
    json!({
      "facts": [],
      "background_snippets": ["I am an engineer."]
    }),
  );
  // The first response is discarded after the simulated concurrent update; the
  // retry consumes the second.
  llm.push(
    "background_merge",
    json!({ "background": "I am an engineer." }),
  );
  llm.push(
    "background_merge",
    json!({ "background": "I am an engineer." }),
  );

  let racing: Arc<dyn Llm> = Arc::new(RacingLlm {
    inner: llm.clone(),
    pool: sqlite.pool().clone(),
    bank,
    raced: AtomicBool::new(false),
  });
  let memory = MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(1_000))),
    RetainConfig::default(),
  )
  .with_models(racing, Arc::new(HashEmbedder::new(EMBEDDING_DIM)))
  .with_worker_config(WorkerConfig {
    max_attempts: 1,
    ..WorkerConfig::default()
  });

  let source = memory
    .retain(bank, payload("I am an engineer."))
    .await
    .expect("retain");
  let report = memory.rescan(bank).await.expect("rescan");

  assert_eq!(report.chunks_processed, 1);
  assert_eq!(report.failed, 0);

  let (background, updated_at): (String, i64) =
    sqlx::query_as("SELECT background, updated_at FROM bank WHERE id = ?")
      .bind(bank.to_string())
      .fetch_one(sqlite.pool())
      .await
      .expect("bank");
  assert_eq!(background, "I am an engineer.");
  assert_eq!(updated_at, 1_000);

  let status: String = sqlx::query_scalar("SELECT status FROM chunk WHERE source_id = ?")
    .bind(source.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("chunk");
  assert_eq!(status, "done");

  let merge_calls: Vec<_> = llm
    .calls()
    .into_iter()
    .filter(|call| call.schema_name == "background_merge")
    .collect();
  assert_eq!(
    merge_calls.len(),
    2,
    "the CAS conflict triggers one re-merge"
  );
  assert!(
    merge_calls[1]
      .user
      .contains("CURRENT BACKGROUND:\nI am from a concurrent chunk."),
    "the retry re-reads the winning background"
  );
}

#[tokio::test]
async fn a_failed_tx2_background_cas_rolls_back_all_products() {
  let (sqlite, store, bank, llm) = setup(1_000).await;
  llm.push(
    "fact_extraction",
    json!({
      "facts": [fact_json("Alice met Bob", "Alice")],
      "background_snippets": ["I am an engineer."]
    }),
  );
  llm.push(
    "background_merge",
    json!({ "background": "I am an engineer." }),
  );

  let racing: Arc<dyn Llm> = Arc::new(AlwaysRacingLlm {
    inner: llm.clone(),
    pool: sqlite.pool().clone(),
    bank,
  });
  let pipeline = PipelineConfig {
    background: BackgroundConfig {
      max_attempts: 1,
      ..BackgroundConfig::default()
    },
    ..PipelineConfig::default()
  };
  let memory = MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(1_000))),
    RetainConfig::default(),
  )
  .with_models(racing, Arc::new(HashEmbedder::new(EMBEDDING_DIM)))
  .with_pipeline(pipeline)
  .with_worker_config(WorkerConfig {
    max_attempts: 1,
    ..WorkerConfig::default()
  });

  let source = memory
    .retain(bank, payload("Alice met Bob. I am an engineer."))
    .await
    .expect("retain");
  let report = memory.rescan(bank).await.expect("rescan");
  assert_eq!(report.chunks_processed, 0);
  assert_eq!(report.failed, 1);

  let counts: [i64; 5] = [
    sqlx::query_scalar("SELECT COUNT(*) FROM fact")
      .fetch_one(sqlite.pool())
      .await
      .expect("facts"),
    sqlx::query_scalar("SELECT COUNT(*) FROM fact_vec")
      .fetch_one(sqlite.pool())
      .await
      .expect("vectors"),
    sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity")
      .fetch_one(sqlite.pool())
      .await
      .expect("bindings"),
    sqlx::query_scalar("SELECT COUNT(*) FROM fact_fts")
      .fetch_one(sqlite.pool())
      .await
      .expect("fts"),
    sqlx::query_scalar("SELECT COUNT(*) FROM fact_link")
      .fetch_one(sqlite.pool())
      .await
      .expect("links"),
  ];
  for count in counts {
    assert_eq!(count, 0, "the rolled-back tx2 left no products");
  }
  let entities: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entity")
    .fetch_one(sqlite.pool())
    .await
    .expect("entities");
  assert_eq!(entities, 0);

  let background: String = sqlx::query_scalar("SELECT background FROM bank WHERE id = ?")
    .bind(bank.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("bank");
  assert_eq!(
    background, "I am from a racing update.",
    "the failed merge did not overwrite the concurrent update"
  );

  let status = memory.retain_status(bank, source).await.expect("status");
  assert_eq!(status.state, RetainState::Failed);
}
