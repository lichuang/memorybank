//! P1-13: semantic link construction against a real store.

use engine::link::{LinkConfig, build_semantic_links};
use engine::ports::{NewLink, Store};
use engine::testing::timestamp;
use engine::types::{
  Bank, BankId, BankProfile, Fact, FactId, FactType, LinkType, OccurrenceInterval,
};
use sqlx::SqlitePool;
use store::testing::memory_pool;
use store::{EMBEDDING_DIM, SqliteStore};

fn ts(millis: i64) -> engine::types::Timestamp {
  timestamp(millis)
}

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
    created_at: ts(0),
    updated_at: ts(0),
  };
  store.create_bank(bank.clone()).await.expect("create bank");
  (store, bank.id)
}

fn fact(bank: BankId, embedding: Option<Vec<f32>>) -> Fact {
  Fact {
    id: FactId::new(),
    bank_id: bank,
    fact_type: FactType::World,
    text: "fact".into(),
    embedding,
    chunk_id: None,
    occurred: OccurrenceInterval::Unknown,
    mentioned_at: None,
    confidence: None,
    access_count: 0,
    attrs: None,
    created_at: ts(0),
    updated_at: ts(0),
  }
}

fn blob(values: &[f32]) -> Vec<u8> {
  values
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect()
}

/// Builds a sparse `EMBEDDING_DIM`-dimensional test vector.
fn vector(components: &[(usize, f32)]) -> Vec<f32> {
  let mut values = vec![0.0; EMBEDDING_DIM];
  for (index, value) in components {
    values[*index] = *value;
  }
  values
}

async fn insert_existing(pool: &SqlitePool, fact: &Fact) {
  // fact and fact_vec are 1:1; the production write path commits them in one
  // transaction (tx2), so the fixture does the same.
  let mut tx = pool.begin().await.expect("begin insert");

  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, created_at, updated_at) \
     VALUES (?, ?, 'world', ?, 0, 0)",
  )
  .bind(fact.id.to_string())
  .bind(fact.bank_id.to_string())
  .bind(&fact.text)
  .execute(&mut *tx)
  .await
  .expect("insert fact");

  if let Some(embedding) = fact.embedding.as_deref() {
    sqlx::query(
      "INSERT INTO fact_vec(rowid, fact_id, bank_id, embedding) \
       SELECT rowid, id, bank_id, ? FROM fact WHERE id = ?",
    )
    .bind(blob(embedding))
    .bind(fact.id.to_string())
    .execute(&mut *tx)
    .await
    .expect("insert vector");
  }

  tx.commit().await.expect("commit insert");
}

fn config(top_k: usize, threshold: f32) -> LinkConfig {
  LinkConfig {
    semantic_top_k: top_k,
    semantic_threshold: threshold,
    ..LinkConfig::default()
  }
}

fn pairs(links: &[NewLink]) -> Vec<(FactId, FactId)> {
  let mut pairs: Vec<(FactId, FactId)> = links.iter().map(|link| (link.from, link.to)).collect();
  pairs.sort();
  pairs
}

#[tokio::test]
async fn links_to_similar_existing_facts_as_directed_rows() {
  let (store, bank) = setup().await;
  let similar = fact(bank, Some(vector(&[(0, 1.0)])));
  let orthogonal = fact(bank, Some(vector(&[(1, 1.0)])));
  insert_existing(store.pool(), &similar).await;
  insert_existing(store.pool(), &orthogonal).await;
  let new = fact(bank, Some(vector(&[(0, 1.0)])));

  let links = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(5, 0.8),
    ts(1),
  )
  .await
  .expect("build");

  assert_eq!(links.len(), 1, "only the similar fact passes the threshold");
  assert_eq!(links[0].from, new.id);
  assert_eq!(links[0].to, similar.id);
  assert_eq!(links[0].link_type, LinkType::Semantic);
  assert!(links[0].relation.is_none());
  assert_eq!(links[0].created_at, ts(1));
  assert!((links[0].weight - 1.0).abs() < 1e-6);
  assert!(
    !links.iter().any(|link| link.from == similar.id),
    "an existing fact never selects back in this pass"
  );
}

#[tokio::test]
async fn threshold_filters_weak_candidates() {
  let (store, bank) = setup().await;
  let diagonal = fact(bank, Some(vector(&[(0, 1.0), (1, 1.0)])));
  insert_existing(store.pool(), &diagonal).await;
  let new = fact(bank, Some(vector(&[(0, 1.0)])));

  let strict = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(5, 0.8),
    ts(1),
  )
  .await
  .expect("strict");
  assert!(strict.is_empty(), "cosine ≈ 0.707 is below 0.8");

  let loose = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(5, 0.7),
    ts(1),
  )
  .await
  .expect("loose");
  assert_eq!(loose.len(), 1);
  assert!((loose[0].weight - 0.707_106_77).abs() < 1e-6);
}

#[tokio::test]
async fn budget_keeps_the_most_similar_candidates() {
  let (store, bank) = setup().await;
  let closest = fact(bank, Some(vector(&[(0, 1.0), (1, 0.1)])));
  let second = fact(bank, Some(vector(&[(0, 1.0), (1, 0.2)])));
  let third = fact(bank, Some(vector(&[(0, 1.0), (1, 0.3)])));
  for existing in [&closest, &second, &third] {
    insert_existing(store.pool(), existing).await;
  }
  let new = fact(bank, Some(vector(&[(0, 1.0)])));

  let links = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(2, 0.5),
    ts(1),
  )
  .await
  .expect("build");

  let mut expected = vec![(new.id, closest.id), (new.id, second.id)];
  expected.sort();
  assert_eq!(pairs(&links), expected);
}

#[tokio::test]
async fn new_new_pairs_are_linked_directionally() {
  let (store, bank) = setup().await;
  let a = fact(bank, Some(vector(&[(0, 1.0)])));
  let b = fact(bank, Some(vector(&[(0, 1.0)])));

  let links = build_semantic_links(
    &store,
    bank,
    &[a.clone(), b.clone()],
    &config(5, 0.8),
    ts(1),
  )
  .await
  .expect("build");

  let mut expected = vec![(a.id, b.id), (b.id, a.id)];
  expected.sort();
  assert_eq!(
    pairs(&links),
    expected,
    "each new fact selects the other; two independent directed rows"
  );
}

#[tokio::test]
async fn a_fact_never_links_to_itself() {
  let (store, bank) = setup().await;
  let new = fact(bank, Some(vector(&[(0, 1.0)])));
  // Simulate a re-run where the fact is already committed.
  insert_existing(store.pool(), &new).await;

  let links = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(5, 0.8),
    ts(1),
  )
  .await
  .expect("build");

  assert!(links.is_empty());
}

#[tokio::test]
async fn facts_without_embeddings_create_no_edges() {
  let (store, bank) = setup().await;
  let existing = fact(bank, Some(vector(&[(0, 1.0)])));
  insert_existing(store.pool(), &existing).await;
  let new = fact(bank, None);

  let links = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(5, 0.8),
    ts(1),
  )
  .await
  .expect("build");

  assert!(links.is_empty());
}

#[tokio::test]
async fn dimension_mismatch_is_a_config_error() {
  let (store, bank) = setup().await;
  let existing = fact(bank, Some(vector(&[(0, 1.0)])));
  insert_existing(store.pool(), &existing).await;
  let new = fact(bank, Some(vec![1.0, 0.0, 0.0]));

  let error = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(5, 0.8),
    ts(1),
  )
  .await
  .expect_err("dimension mismatch");

  assert!(matches!(error, engine::Error::Config(_)), "error: {error}");
}

#[tokio::test]
async fn rejects_an_out_of_range_threshold_and_allows_disabling() {
  let (store, bank) = setup().await;
  let new = fact(bank, Some(vector(&[(0, 1.0)])));

  let error = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(5, 1.5),
    ts(1),
  )
  .await
  .expect_err("threshold > 1");
  assert!(matches!(error, engine::Error::Config(_)), "error: {error}");

  let disabled = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(0, 0.8),
    ts(1),
  )
  .await
  .expect("disabled");
  assert!(disabled.is_empty());
}

#[tokio::test]
async fn selection_is_deterministic() {
  let (store, bank) = setup().await;
  let existing = fact(bank, Some(vector(&[(0, 1.0)])));
  insert_existing(store.pool(), &existing).await;
  let new = fact(bank, Some(vector(&[(0, 1.0)])));

  let first = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(5, 0.8),
    ts(1),
  )
  .await
  .expect("first");
  let second = build_semantic_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(5, 0.8),
    ts(1),
  )
  .await
  .expect("second");

  assert_eq!(pairs(&first), pairs(&second));
  let first_weights: Vec<f32> = first.iter().map(|link| link.weight).collect();
  let second_weights: Vec<f32> = second.iter().map(|link| link.weight).collect();
  assert_eq!(first_weights, second_weights);
}
