//! End-to-end recall through the facade.

use std::sync::Arc;

use engine::ports::{Embedder, FactScope, Store};
use engine::recall::{Channel, ChannelMask, RecallOptions};
use engine::testing::{FixedClock, HashEmbedder, IdentityReranker, RuleTemporalParser, timestamp};
use engine::types::{Bank, BankId, BankProfile, FactId, FactType};
use memorybank::{MemoryBank, RetainConfig};
use sqlx::SqlitePool;
use store::testing::memory_pool;
use store::{EMBEDDING_DIM, SqliteStore};

fn ts(millis: i64) -> engine::types::Timestamp {
  timestamp(millis)
}

fn hour(value: i64) -> i64 {
  value * 60 * 60 * 1_000
}

async fn setup() -> (SqliteStore, Arc<dyn Store>, BankId) {
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
    created_at: ts(0),
    updated_at: ts(0),
  };
  sqlite.create_bank(bank.clone()).await.expect("bank");
  let store: Arc<dyn Store> = Arc::new(sqlite.clone());
  (sqlite, store, bank.id)
}

fn memory(store: Arc<dyn Store>) -> MemoryBank {
  MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(1_000))),
    RetainConfig::default(),
  )
  .with_models(
    Arc::new(engine::testing::FakeLlm::new()),
    Arc::new(HashEmbedder::new(EMBEDDING_DIM)),
  )
  .with_recall_models(Arc::new(IdentityReranker), Arc::new(RuleTemporalParser))
}

fn blob(values: &[f32]) -> Vec<u8> {
  values
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect()
}

/// Inserts a fact with text, optional embedding, and an optional point interval.
async fn insert_fact(
  pool: &SqlitePool,
  bank: BankId,
  fact_type: FactType,
  text: &str,
  embedding: Option<&[f32]>,
  at: Option<i64>,
) -> FactId {
  let id = FactId::new();
  let confidence = (fact_type == FactType::Opinion).then_some(0.5_f64);

  let mut tx = pool.begin().await.expect("tx");
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, occurred_start, occurred_end, \
     confidence, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(fact_type.as_str())
  .bind(text)
  .bind(at)
  .bind(at)
  .bind(confidence)
  .execute(&mut *tx)
  .await
  .expect("fact");

  sqlx::query("INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)")
    .bind(id.to_string())
    .bind(text)
    .execute(&mut *tx)
    .await
    .expect("fts");

  if let Some(embedding) = embedding {
    sqlx::query(
      "INSERT INTO fact_vec(rowid, fact_id, bank_id, fact_type, embedding) \
       SELECT rowid, id, bank_id, fact_type, ? FROM fact WHERE id = ?",
    )
    .bind(blob(embedding))
    .bind(id.to_string())
    .execute(&mut *tx)
    .await
    .expect("vector");
  }

  tx.commit().await.expect("commit");
  id
}

#[tokio::test]
async fn recall_fuses_channels_and_reports_provenance() {
  let (sqlite, store, bank) = setup().await;
  let query = "Emily works at Google";
  // The embedder hashes text; make the query and one fact share an embedding.
  let embedding = HashEmbedder::new(EMBEDDING_DIM)
    .embed(&[query])
    .await
    .expect("embed")
    .pop()
    .expect("vector");

  let semantic = insert_fact(
    sqlite.pool(),
    bank,
    FactType::World,
    query,
    Some(&embedding),
    Some(0),
  )
  .await;
  // A lexically overlapping fact that is also embedded, so it can surface in
  // both the vector and BM25 lists.
  let lexical_text = "Emily works at Google on search";
  let lexical_embedding = HashEmbedder::new(EMBEDDING_DIM)
    .embed(&[lexical_text])
    .await
    .expect("embed lexical")
    .pop()
    .expect("vector");
  let lexical = insert_fact(
    sqlite.pool(),
    bank,
    FactType::World,
    lexical_text,
    Some(&lexical_embedding),
    None,
  )
  .await;
  let other = insert_fact(
    sqlite.pool(),
    bank,
    FactType::World,
    "Bob likes Rust",
    None,
    None,
  )
  .await;

  let memory = memory(store);
  let options = RecallOptions {
    token_budget: 1_000,
    channels: ChannelMask {
      semantic: true,
      bm25: true,
      graph: false,
      temporal: false,
    },
    per_channel_k: 10,
    rerank_top_n: 0,
    ..RecallOptions::default()
  };

  let result = memory.recall(bank, query, options).await.expect("recall");

  let ids = result.ids();
  assert!(ids.contains(&semantic), "vector hit");
  assert!(ids.contains(&lexical), "BM25 hit");
  assert!(!ids.contains(&other), "unrelated facts stay out");

  assert!(
    result.is_multi_channel(lexical),
    "the shared term makes it a lexical and vector candidate"
  );
  assert!(
    result
      .channels_of(semantic)
      .iter()
      .any(|(channel, _)| *channel == Channel::Semantic)
  );

  let access: i64 = sqlx::query_scalar("SELECT SUM(access_count) FROM fact")
    .fetch_one(sqlite.pool())
    .await
    .expect("access sum");
  assert_eq!(
    access,
    result.facts.len() as i64,
    "each returned fact is counted once"
  );
}

#[tokio::test]
async fn recall_respects_the_token_budget() {
  let (sqlite, store, bank) = setup().await;
  let query = "Emily works at Google";
  let embedding = HashEmbedder::new(EMBEDDING_DIM)
    .embed(&[query])
    .await
    .expect("embed")
    .pop()
    .expect("vector");

  for index in 0..20 {
    insert_fact(
      sqlite.pool(),
      bank,
      FactType::World,
      &format!("Emily works at Google on project {index}"),
      Some(&embedding),
      None,
    )
    .await;
  }

  let memory = memory(store);
  let options = RecallOptions {
    token_budget: 40,
    channels: ChannelMask::only(Channel::Semantic),
    per_channel_k: 20,
    rerank_top_n: 0,
    ..RecallOptions::default()
  };

  let result = memory.recall(bank, query, options).await.expect("recall");

  assert!(!result.facts.is_empty());
  assert!(
    result.tokens_used <= 40,
    "tokens_used {} exceeds the budget",
    result.tokens_used
  );
  assert!(
    result.facts.len() < 20,
    "the budget must actually truncate the list"
  );
}

#[tokio::test]
async fn recall_skips_disabled_channels() {
  let (sqlite, store, bank) = setup().await;
  insert_fact(
    sqlite.pool(),
    bank,
    FactType::World,
    "Emily works at Google",
    None,
    None,
  )
  .await;

  let memory = memory(store);

  let none = memory
    .recall(
      bank,
      "Emily",
      RecallOptions {
        channels: ChannelMask::NONE,
        ..RecallOptions::default()
      },
    )
    .await
    .expect("no channels");
  assert!(none.facts.is_empty());
  assert_eq!(none.tokens_used, 0);

  let bm25_only = memory
    .recall(
      bank,
      "Emily",
      RecallOptions {
        channels: ChannelMask::only(Channel::Bm25),
        rerank_top_n: 0,
        ..RecallOptions::default()
      },
    )
    .await
    .expect("bm25 only");
  assert_eq!(bm25_only.facts.len(), 1);
  assert_eq!(
    bm25_only.facts[0].channels,
    vec![(Channel::Bm25, 1)],
    "the disabled channels contribute no provenance"
  );
}

#[tokio::test]
async fn recall_honors_scope_and_explicit_time_range() {
  let (sqlite, store, bank) = setup().await;
  let opinion = insert_fact(
    sqlite.pool(),
    bank,
    FactType::Opinion,
    "I think Emily is reliable",
    None,
    None,
  )
  .await;
  insert_fact(
    sqlite.pool(),
    bank,
    FactType::World,
    "Emily works at Google",
    None,
    None,
  )
  .await;
  let timed = insert_fact(
    sqlite.pool(),
    bank,
    FactType::World,
    "Emily joined the team",
    None,
    Some(hour(3)),
  )
  .await;

  let memory = memory(store);

  let opinions = memory
    .recall(
      bank,
      "Emily",
      RecallOptions {
        channels: ChannelMask::only(Channel::Bm25),
        scope: FactScope::Types(vec![FactType::Opinion]),
        rerank_top_n: 0,
        ..RecallOptions::default()
      },
    )
    .await
    .expect("scope");
  assert_eq!(opinions.ids(), vec![opinion]);

  let temporal = memory
    .recall(
      bank,
      "irrelevant text",
      RecallOptions {
        channels: ChannelMask::only(Channel::Temporal),
        time_range: Some(engine::types::TimeRange {
          start: ts(hour(2)),
          end: ts(hour(4)),
        }),
        rerank_top_n: 0,
        ..RecallOptions::default()
      },
    )
    .await
    .expect("time range");
  assert_eq!(temporal.ids(), vec![timed]);
}

#[tokio::test]
async fn graph_channel_does_not_leak_facts_outside_the_scope() {
  let (sqlite, store, bank) = setup().await;
  let query = "Emily works at Google";
  let embedding = HashEmbedder::new(EMBEDDING_DIM)
    .embed(&[query])
    .await
    .expect("embed")
    .pop()
    .expect("vector");

  let opinion = insert_fact(
    sqlite.pool(),
    bank,
    FactType::Opinion,
    query,
    Some(&embedding),
    None,
  )
  .await;
  let world = insert_fact(
    sqlite.pool(),
    bank,
    FactType::World,
    "Emily works at Google on search",
    None,
    None,
  )
  .await;

  // The world fact is reachable from the opinion through both a semantic edge
  // and a shared entity, so a scope-blind graph channel would return it.
  sqlx::query(
    "INSERT INTO fact_link (id, bank_id, from_fact_id, to_fact_id, link_type, relation, \
     weight, created_at, updated_at) VALUES (?, ?, ?, ?, 'semantic', NULL, 0.9, 0, 0)",
  )
  .bind(engine::types::LinkId::new().to_string())
  .bind(bank.to_string())
  .bind(opinion.to_string())
  .bind(world.to_string())
  .execute(sqlite.pool())
  .await
  .expect("link");

  let entity = engine::types::EntityId::new();
  sqlx::query(
    "INSERT INTO entity (id, bank_id, canonical_name, entity_type, mention_count, \
     created_at, updated_at) VALUES (?, ?, 'emily', 'person', 0, 0, 0)",
  )
  .bind(entity.to_string())
  .bind(bank.to_string())
  .execute(sqlite.pool())
  .await
  .expect("entity");
  for fact in [opinion, world] {
    sqlx::query(
      "INSERT INTO fact_entity (bank_id, fact_id, entity_id, mention_text, created_at, updated_at) \
       VALUES (?, ?, ?, NULL, 0, 0)",
    )
    .bind(bank.to_string())
    .bind(fact.to_string())
    .bind(entity.to_string())
    .execute(sqlite.pool())
    .await
    .expect("binding");
  }

  let memory = memory(store);
  let options = RecallOptions {
    channels: ChannelMask::only(Channel::Graph),
    scope: FactScope::Types(vec![FactType::Opinion]),
    rerank_top_n: 0,
    ..RecallOptions::default()
  };
  let result = memory.recall(bank, query, options).await.expect("recall");

  // Graph requires semantic seeds; with only the graph mask enabled there are
  // none, so the channel is skipped and nothing is returned.
  assert!(result.facts.is_empty(), "ids: {:?}", result.ids());

  // Now run the real combination: semantic seeds + graph + opinion scope.
  let options = RecallOptions {
    channels: ChannelMask {
      semantic: true,
      bm25: false,
      graph: true,
      temporal: false,
    },
    scope: FactScope::Types(vec![FactType::Opinion]),
    rerank_top_n: 0,
    ..RecallOptions::default()
  };
  let result = memory
    .recall(bank, query, options)
    .await
    .expect("recall with seeds");

  assert_eq!(
    result.ids(),
    vec![opinion],
    "the world fact is filtered out"
  );
}

#[tokio::test]
async fn recall_returns_empty_for_an_empty_bank() {
  let (_sqlite, store, bank) = setup().await;
  let memory = memory(store);

  let result = memory
    .recall(bank, "anything", RecallOptions::default())
    .await
    .expect("empty bank");
  assert!(result.facts.is_empty());
  assert_eq!(result.tokens_used, 0);
}

#[tokio::test]
async fn recall_requires_an_embedder() {
  let (_, store, bank) = setup().await;
  let memory = MemoryBank::new(
    store,
    Arc::new(FixedClock(ts(1_000))),
    RetainConfig::default(),
  )
  .with_recall_models(Arc::new(IdentityReranker), Arc::new(RuleTemporalParser));

  let error = memory
    .recall(bank, "query", RecallOptions::default())
    .await
    .expect_err("no embedder");
  assert!(matches!(error, engine::Error::Config(_)), "error: {error}");
}
