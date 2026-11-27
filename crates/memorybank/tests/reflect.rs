//! End-to-end reflect through the facade.

use std::sync::Arc;

use engine::ports::{Embedder, Store};
use engine::recall::{ChannelMask, RecallOptions};
use engine::reflect::SystemMessageConfig;
use engine::testing::{FakeLlm, FixedClock, HashEmbedder, IdentityReranker, RuleTemporalParser};
use engine::types::{Bank, BankId, BankProfile, FactId};
use memorybank::{MemoryBank, ReflectOptions, RetainConfig};
use serde_json::json;
use sqlx::SqlitePool;
use store::testing::memory_pool;
use store::{EMBEDDING_DIM, SqliteStore};

fn ts(millis: i64) -> engine::types::Timestamp {
  engine::testing::timestamp(millis)
}

async fn setup() -> (SqliteStore, Arc<dyn Store>, BankId, Arc<FakeLlm>) {
  let sqlite = SqliteStore::new(memory_pool().await.expect("pool"));
  let bank = Bank {
    id: BankId::new(),
    name: "primary".into(),
    profile: BankProfile {
      background: "I am a careful assistant.".into(),
      skepticism: 2,
      literalism: 2,
      empathy: 4,
      bias_strength: 0.5,
    },
    created_at: ts(0),
    updated_at: ts(0),
  };
  sqlite.create_bank(bank.clone()).await.expect("bank");
  let store: Arc<dyn Store> = Arc::new(sqlite.clone());
  (sqlite, store, bank.id, Arc::new(FakeLlm::new()))
}

fn memory(store: Arc<dyn Store>, llm: Arc<FakeLlm>, now: i64) -> MemoryBank {
  let llm: Arc<dyn engine::ports::Llm> = llm;
  MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(now))),
    RetainConfig::default(),
  )
  .with_models(llm, Arc::new(HashEmbedder::new(EMBEDDING_DIM)))
  .with_recall_models(Arc::new(IdentityReranker), Arc::new(RuleTemporalParser))
}

async fn embedding(text: &str) -> Vec<f32> {
  HashEmbedder::new(EMBEDDING_DIM)
    .embed(&[text])
    .await
    .expect("embed")
    .pop()
    .expect("vector")
}

fn blob(values: &[f32]) -> Vec<u8> {
  values
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect()
}

async fn insert_fact(pool: &SqlitePool, bank: BankId, text: &str, with_embedding: bool) -> FactId {
  let id = FactId::new();
  let mut tx = pool.begin().await.expect("tx");
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, created_at, updated_at) \
     VALUES (?, ?, 'world', ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(text)
  .execute(&mut *tx)
  .await
  .expect("fact");

  sqlx::query("INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)")
    .bind(id.to_string())
    .bind(text)
    .execute(&mut *tx)
    .await
    .expect("fts");

  if with_embedding {
    let vector = embedding(text).await;
    sqlx::query(
      "INSERT INTO fact_vec(rowid, fact_id, bank_id, fact_type, embedding) \
       SELECT rowid, id, bank_id, fact_type, ? FROM fact WHERE id = ?",
    )
    .bind(blob(&vector))
    .bind(id.to_string())
    .execute(&mut *tx)
    .await
    .expect("vector");
  }

  tx.commit().await.expect("commit");
  id
}

fn options() -> ReflectOptions {
  ReflectOptions {
    recall: RecallOptions {
      channels: ChannelMask {
        semantic: true,
        bm25: true,
        graph: false,
        temporal: false,
      },
      per_channel_k: 10,
      rerank_top_n: 0,
      ..RecallOptions::default()
    },
    message: SystemMessageConfig::default(),
    ..ReflectOptions::default()
  }
}

#[tokio::test]
async fn reflect_answers_with_evidence_and_persists_opinions() {
  let (sqlite, store, bank, llm) = setup().await;
  let first = insert_fact(sqlite.pool(), bank, "Alice ships reliably", true).await;
  insert_fact(sqlite.pool(), bank, "Bob reviews designs carefully", true).await;

  llm.push(
    "reflect",
    json!({
      "answer": "Alice ships reliably [1].",
      "opinions": [
        { "opinion": "I think Alice is dependable", "confidence": 0.7, "reasoning": "delivery record" }
      ]
    }),
  );

  let memory = memory(store, llm.clone(), 1_000);
  let result = memory
    .reflect(bank, "Is Alice dependable?", options())
    .await
    .expect("reflect");

  assert_eq!(result.answer, "Alice ships reliably [1].");
  assert!(result.has_evidence());
  assert_eq!(result.evidence_id(1), Some(first));
  assert_eq!(result.opinions_formed.len(), 1);

  let opinion_id = result.opinions_formed[0];
  let (fact_type, chunk_id, confidence): (String, Option<String>, f64) =
    sqlx::query_as("SELECT fact_type, chunk_id, confidence FROM fact WHERE id = ?")
      .bind(opinion_id.to_string())
      .fetch_one(sqlite.pool())
      .await
      .expect("opinion");
  assert_eq!(fact_type, "opinion");
  assert_eq!(chunk_id, None);
  assert!((confidence - 0.7).abs() < 1e-6);

  let vectors: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_vec WHERE fact_id = ?")
    .bind(opinion_id.to_string())
    .fetch_one(sqlite.pool())
    .await
    .expect("vector");
  assert_eq!(vectors, 1, "the opinion is embedded for later recall");
}

#[tokio::test]
async fn an_empty_opinion_list_writes_nothing() {
  let (sqlite, store, bank, llm) = setup().await;
  insert_fact(sqlite.pool(), bank, "Alice ships reliably", true).await;

  llm.push(
    "reflect",
    json!({ "answer": "I do not have enough information.", "opinions": [] }),
  );

  let memory = memory(store, llm.clone(), 1_000);
  let result = memory
    .reflect(bank, "Is Alice dependable?", options())
    .await
    .expect("reflect");

  assert!(result.opinions_formed.is_empty());
  let opinions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact WHERE fact_type = 'opinion'")
    .fetch_one(sqlite.pool())
    .await
    .expect("opinions");
  assert_eq!(opinions, 0);
}

#[tokio::test]
async fn persistence_can_be_disabled() {
  let (sqlite, store, bank, llm) = setup().await;
  insert_fact(sqlite.pool(), bank, "Alice ships reliably", true).await;

  llm.push(
    "reflect",
    json!({
      "answer": "Alice ships reliably [1].",
      "opinions": [{ "opinion": "I think Alice is dependable", "confidence": 0.7, "reasoning": "r" }]
    }),
  );

  let memory = memory(store, llm.clone(), 1_000);
  let mut opts = options();
  opts.persist_opinions = false;
  let result = memory
    .reflect(bank, "Is Alice dependable?", opts)
    .await
    .expect("reflect");

  assert!(result.opinions_formed.is_empty());
  let opinions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact WHERE fact_type = 'opinion'")
    .fetch_one(sqlite.pool())
    .await
    .expect("opinions");
  assert_eq!(opinions, 0);
}

#[tokio::test]
async fn reflect_requires_an_llm() {
  let (_, store, bank, _llm) = setup().await;
  let memory = MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(1_000))),
    RetainConfig::default(),
  );

  let error = memory
    .reflect(bank, "query", options())
    .await
    .expect_err("no llm");
  assert!(matches!(error, engine::Error::Config(_)), "error: {error}");
}

#[tokio::test]
async fn an_empty_query_is_rejected_before_any_call() {
  let (_sqlite, store, bank, llm) = setup().await;
  let memory = memory(store, llm.clone(), 1_000);

  let error = memory
    .reflect(bank, "   ", options())
    .await
    .expect_err("empty query");
  assert!(matches!(error, engine::Error::Config(_)), "error: {error}");
  assert!(llm.calls().is_empty(), "no model call for an empty query");
}

#[tokio::test]
async fn a_failed_reflect_call_writes_no_opinion() {
  let (sqlite, store, bank, llm) = setup().await;
  insert_fact(sqlite.pool(), bank, "Alice ships reliably", true).await;

  // No scripted response: the fake reports a deterministic failure.
  let memory = memory(store, llm.clone(), 1_000);
  let error = memory
    .reflect(bank, "Is Alice dependable?", options())
    .await
    .expect_err("llm failure");
  assert!(!error.is_retryable(), "error: {error}");

  let opinions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact WHERE fact_type = 'opinion'")
    .fetch_one(sqlite.pool())
    .await
    .expect("opinions");
  assert_eq!(opinions, 0);
}

#[tokio::test]
async fn reflect_records_access_on_the_evidence() {
  let (sqlite, store, bank, llm) = setup().await;
  insert_fact(sqlite.pool(), bank, "Alice ships reliably", true).await;

  llm.push(
    "reflect",
    json!({ "answer": "Alice ships reliably [1].", "opinions": [] }),
  );

  let memory = memory(store, llm.clone(), 1_000);
  let result = memory
    .reflect(bank, "Is Alice dependable?", options())
    .await
    .expect("reflect");

  let access: i64 =
    sqlx::query_scalar("SELECT SUM(access_count) FROM fact WHERE fact_type = 'world'")
      .fetch_one(sqlite.pool())
      .await
      .expect("access");
  assert_eq!(access, result.recalled.facts.len() as i64);
}

#[tokio::test]
async fn the_system_prompt_carries_the_profile_and_evidence() {
  let (sqlite, store, bank, llm) = setup().await;
  insert_fact(sqlite.pool(), bank, "Alice ships reliably", true).await;

  llm.push(
    "reflect",
    json!({ "answer": "Alice ships reliably [1].", "opinions": [] }),
  );

  let memory = memory(store, llm.clone(), 1_000);
  memory
    .reflect(bank, "Is Alice dependable?", options())
    .await
    .expect("reflect");

  let calls = llm.calls();
  let system = calls[0].system.as_deref().expect("system");
  assert!(system.contains("bank \"primary\""));
  assert!(system.contains("mostly trusting"));
  assert!(system.contains("[1] Alice ships reliably"));
  assert!(calls[0].user.contains("Is Alice dependable?"));
}
