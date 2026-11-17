//! BM25 recall channel over the FTS5 trigram index.

use engine::ports::{FactScope, Store};
use engine::recall::{Bm25Config, bm25_channel};
use engine::types::{Bank, BankId, BankProfile, FactId, FactType};
use sqlx::SqlitePool;
use store::SqliteStore;
use store::testing::memory_pool;

async fn setup() -> (SqliteStore, BankId) {
  let store = SqliteStore::new(memory_pool().await.expect("pool"));
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
    created_at: engine::testing::timestamp(0),
    updated_at: engine::testing::timestamp(0),
  };
  store.create_bank(bank.clone()).await.expect("create bank");
  (store, bank.id)
}

async fn insert_fact(pool: &SqlitePool, bank: BankId, fact_type: FactType, text: &str) -> FactId {
  let id = FactId::new();
  let confidence = (fact_type == FactType::Opinion).then_some(0.5_f64);

  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, confidence, created_at, updated_at) \
     VALUES (?, ?, ?, ?, ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(fact_type.as_str())
  .bind(text)
  .bind(confidence)
  .execute(pool)
  .await
  .expect("fact");

  sqlx::query("INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)")
    .bind(id.to_string())
    .bind(text)
    .execute(pool)
    .await
    .expect("fts");

  id
}

async fn search(
  store: &SqliteStore,
  bank: BankId,
  query: &str,
  top_k: usize,
  scope: &FactScope,
) -> Vec<(FactId, f32)> {
  bm25_channel(store, bank, query, &Bm25Config { top_k }, scope)
    .await
    .expect("bm25")
}

#[tokio::test]
async fn matches_proper_nouns_and_numbers_in_ranked_order() {
  let (store, bank) = setup().await;
  let dense = insert_fact(
    store.pool(),
    bank,
    FactType::World,
    "Emily Emily Emily visited Google in 2024",
  )
  .await;
  let sparse = insert_fact(
    store.pool(),
    bank,
    FactType::World,
    "Emily visited Google in 2024",
  )
  .await;
  insert_fact(
    store.pool(),
    bank,
    FactType::World,
    "Alice visited RustConf in 2025",
  )
  .await;

  let hits = search(&store, bank, "Emily", 10, &FactScope::All).await;

  assert_eq!(
    hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![dense, sparse],
    "higher term frequency ranks first"
  );
  assert!(hits.iter().all(|(_, score)| *score > 0.0));

  let google = search(&store, bank, "Google 2024", 10, &FactScope::All).await;
  let ids: Vec<FactId> = google.iter().map(|(id, _)| *id).collect();
  assert!(ids.contains(&dense));
  assert!(ids.contains(&sparse));
}

#[tokio::test]
async fn supports_cjk_trigram_substrings() {
  let (store, bank) = setup().await;
  let beijing = insert_fact(store.pool(), bank, FactType::World, "北京欢迎你").await;
  insert_fact(store.pool(), bank, FactType::World, "上海欢迎你").await;

  let hits = search(&store, bank, "北京欢迎", 10, &FactScope::All).await;
  assert_eq!(
    hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![beijing]
  );

  let too_short = search(&store, bank, "北京", 10, &FactScope::All).await;
  assert!(
    too_short.is_empty(),
    "two-character terms cannot match a trigram index"
  );
}

#[tokio::test]
async fn special_characters_do_not_error() {
  let (store, bank) = setup().await;
  let fact = insert_fact(store.pool(), bank, FactType::World, "Emily joined Google").await;

  for query in [
    "Emily\" OR *",
    "Emily - Google",
    "%%",
    "AND OR NOT",
    "()[]{}",
  ] {
    let hits = search(&store, bank, query, 10, &FactScope::All).await;
    if query == "Emily\" OR *" || query == "Emily - Google" {
      assert!(
        hits.iter().any(|(id, _)| *id == fact),
        "query {query:?} should still match"
      );
    } else {
      assert!(hits.is_empty(), "query {query:?} should be empty");
    }
  }
}

#[tokio::test]
async fn respects_scope_bank_and_top_k() {
  let (store, bank) = setup().await;
  let world = insert_fact(store.pool(), bank, FactType::World, "Emily works on Rust").await;
  let opinion = insert_fact(store.pool(), bank, FactType::Opinion, "Emily likes Rust").await;

  let other_bank = Bank {
    id: BankId::new(),
    name: "other".into(),
    profile: BankProfile {
      background: "background".into(),
      skepticism: 3,
      literalism: 3,
      empathy: 3,
      bias_strength: 0.5,
    },
    created_at: engine::testing::timestamp(0),
    updated_at: engine::testing::timestamp(0),
  };
  store.create_bank(other_bank.clone()).await.expect("bank");
  insert_fact(
    store.pool(),
    other_bank.id,
    FactType::World,
    "Emily elsewhere",
  )
  .await;

  let worlds = search(
    &store,
    bank,
    "Emily",
    10,
    &FactScope::Types(vec![FactType::World]),
  )
  .await;
  assert_eq!(
    worlds.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![world]
  );

  let opinions = search(
    &store,
    bank,
    "Emily",
    10,
    &FactScope::Types(vec![FactType::Opinion]),
  )
  .await;
  assert_eq!(
    opinions.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![opinion]
  );

  let limited = search(&store, bank, "Emily", 1, &FactScope::All).await;
  assert_eq!(limited.len(), 1, "top_k is honored");
}

#[tokio::test]
async fn empty_queries_and_short_terms_return_empty() {
  let (store, bank) = setup().await;
  insert_fact(store.pool(), bank, FactType::World, "AI research").await;

  assert!(
    search(&store, bank, "", 10, &FactScope::All)
      .await
      .is_empty()
  );
  assert!(
    search(&store, bank, "   ", 10, &FactScope::All)
      .await
      .is_empty()
  );
  assert!(
    search(&store, bank, "AI", 10, &FactScope::All)
      .await
      .is_empty(),
    "two-character terms are ignored"
  );
  assert!(
    search(&store, bank, "%%", 10, &FactScope::All)
      .await
      .is_empty()
  );

  let hits = search(&store, bank, "AI research", 10, &FactScope::All).await;
  assert_eq!(hits.len(), 1, "the longer term still matches");

  let disabled = search(&store, bank, "research", 0, &FactScope::All).await;
  assert!(disabled.is_empty());
}

#[tokio::test]
async fn empty_types_scope_returns_empty() {
  let (store, bank) = setup().await;
  insert_fact(store.pool(), bank, FactType::World, "Emily").await;

  let hits = search(&store, bank, "Emily", 10, &FactScope::Types(Vec::new())).await;
  assert!(hits.is_empty());
}
