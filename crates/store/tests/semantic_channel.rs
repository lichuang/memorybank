//! Semantic recall channel over the real sqlite-vec index.

use engine::ports::{FactScope, Store};
use engine::recall::{SemanticConfig, semantic_channel};
use engine::types::{Bank, BankId, BankProfile, FactId, FactType};
use sqlx::{Sqlite, SqlitePool, Transaction};
use store::testing::memory_pool;
use store::{EMBEDDING_DIM, SqliteStore};

fn blob(values: &[f32]) -> Vec<u8> {
  values
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect()
}

fn vector(components: &[(usize, f32)]) -> Vec<f32> {
  let mut values = vec![0.0; EMBEDDING_DIM];
  for (index, value) in components {
    values[*index] = *value;
  }
  values
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
    created_at: engine::testing::timestamp(0),
    updated_at: engine::testing::timestamp(0),
  };
  store.create_bank(bank.clone()).await.expect("create bank");
  (store, bank.id)
}

async fn insert_embedding(
  tx: &mut Transaction<'_, Sqlite>,
  id: FactId,
  embedding: &[f32],
) -> Result<(), sqlx::Error> {
  sqlx::query(
    "INSERT INTO fact_vec(rowid, fact_id, bank_id, fact_type, embedding) \
     SELECT rowid, id, bank_id, fact_type, ? FROM fact WHERE id = ?",
  )
  .bind(blob(embedding))
  .bind(id.to_string())
  .execute(&mut **tx)
  .await
  .map(|_| ())
}

async fn insert_fact(
  pool: &SqlitePool,
  bank: BankId,
  fact_type: FactType,
  embedding: Option<&[f32]>,
) -> FactId {
  let id = FactId::new();
  let confidence = (fact_type == FactType::Opinion).then_some(0.5_f64);

  let mut tx = pool.begin().await.expect("tx");
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, confidence, created_at, updated_at) \
     VALUES (?, ?, ?, 'fact', ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(fact_type.as_str())
  .bind(confidence)
  .execute(&mut *tx)
  .await
  .expect("fact");
  if let Some(embedding) = embedding {
    insert_embedding(&mut tx, id, embedding)
      .await
      .expect("vector");
  }
  tx.commit().await.expect("commit");
  id
}

#[tokio::test]
async fn orders_results_and_drops_scores_below_min_cos() {
  let (store, bank) = setup().await;
  let query = vector(&[(0, 1.0)]);
  let first = insert_fact(&store.pool().clone(), bank, FactType::World, Some(&query)).await;
  let second = insert_fact(
    &store.pool().clone(),
    bank,
    FactType::World,
    Some(&vector(&[(0, 0.8), (1, 0.6)])),
  )
  .await;
  let third = insert_fact(
    &store.pool().clone(),
    bank,
    FactType::World,
    Some(&vector(&[(1, 1.0)])),
  )
  .await;

  let config = SemanticConfig {
    top_k: 10,
    min_cos: 0.5,
  };
  let hits = semantic_channel(&store, bank, &query, &config, &FactScope::All)
    .await
    .expect("semantic");

  assert_eq!(
    hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![first, second],
    "third is below the floor and the order is preserved"
  );
  assert!((hits[0].1 - 1.0).abs() < 1e-6);
  assert!((hits[1].1 - 0.8).abs() < 1e-6);
  assert!(!hits.iter().any(|(id, _)| *id == third));
}

#[tokio::test]
async fn applies_top_k_and_inclusive_min_cos() {
  let (store, bank) = setup().await;
  let query = vector(&[(0, 1.0)]);
  let first = insert_fact(&store.pool().clone(), bank, FactType::World, Some(&query)).await;
  let second = insert_fact(
    &store.pool().clone(),
    bank,
    FactType::World,
    Some(&vector(&[(0, 0.8), (1, 0.6)])),
  )
  .await;

  let top_one = SemanticConfig {
    top_k: 1,
    min_cos: 0.0,
  };
  let hits = semantic_channel(&store, bank, &query, &top_one, &FactScope::All)
    .await
    .expect("top one");
  assert_eq!(
    hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![first]
  );

  // Use the stored f32 score itself as the floor: sqlite-vec/FTS round-trips
  // make a literal 0.8 slightly smaller than the returned value.
  let unscored = SemanticConfig {
    top_k: 10,
    min_cos: -1.0,
  };
  let raw = semantic_channel(&store, bank, &query, &unscored, &FactScope::All)
    .await
    .expect("raw");
  let second_score = raw
    .iter()
    .find(|(id, _)| *id == second)
    .map(|(_, score)| *score)
    .expect("second score");

  let inclusive = SemanticConfig {
    top_k: 10,
    min_cos: second_score,
  };
  let hits = semantic_channel(&store, bank, &query, &inclusive, &FactScope::All)
    .await
    .expect("inclusive");
  assert_eq!(
    hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![first, second],
    "a score exactly on min_cos is kept"
  );
}

#[tokio::test]
async fn respects_fact_scope_and_skips_missing_embeddings() {
  let (store, bank) = setup().await;
  let query = vector(&[(0, 1.0)]);
  let world = insert_fact(&store.pool().clone(), bank, FactType::World, Some(&query)).await;
  let opinion = insert_fact(&store.pool().clone(), bank, FactType::Opinion, Some(&query)).await;
  insert_fact(&store.pool().clone(), bank, FactType::World, None).await;

  let config = SemanticConfig {
    top_k: 10,
    min_cos: 0.0,
  };
  let worlds = semantic_channel(
    &store,
    bank,
    &query,
    &config,
    &FactScope::Types(vec![FactType::World]),
  )
  .await
  .expect("world scope");
  assert_eq!(
    worlds.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![world]
  );

  let opinions = semantic_channel(
    &store,
    bank,
    &query,
    &config,
    &FactScope::Types(vec![FactType::Opinion]),
  )
  .await
  .expect("opinion scope");
  assert_eq!(
    opinions.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![opinion]
  );

  let all = semantic_channel(&store, bank, &query, &config, &FactScope::All)
    .await
    .expect("all");
  assert_eq!(all.len(), 2, "the embedding-less fact is skipped");
}

#[tokio::test]
async fn empty_bank_and_zero_top_k_return_empty() {
  let (store, bank) = setup().await;
  let query = vector(&[(0, 1.0)]);
  let config = SemanticConfig {
    top_k: 10,
    min_cos: 0.0,
  };

  let hits = semantic_channel(&store, bank, &query, &config, &FactScope::All)
    .await
    .expect("empty bank");
  assert!(hits.is_empty());

  let disabled = SemanticConfig { top_k: 0, ..config };
  let hits = semantic_channel(&store, bank, &query, &disabled, &FactScope::All)
    .await
    .expect("disabled");
  assert!(hits.is_empty());
}
