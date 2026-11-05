//! P1-13: brute-force vector search used by semantic link construction.

use engine::ports::Store;
use engine::types::BankId;
use sqlx::SqlitePool;
use store::SqliteStore;
use store::testing::memory_pool;

fn blob(values: &[f32]) -> Vec<u8> {
  values
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect()
}

async fn insert_fact(
  pool: &SqlitePool,
  bank: BankId,
  embedding: Option<&[f32]>,
) -> engine::types::FactId {
  let id = engine::types::FactId::new();
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, embedding, created_at, updated_at) \
     VALUES (?, ?, 'world', 'fact', ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(embedding.map(blob))
  .execute(pool)
  .await
  .expect("insert fact");

  id
}

#[tokio::test]
async fn ranks_by_cosine_and_applies_the_limit() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();

  let far = insert_fact(&pool, bank, Some(&[-1.0, 0.0])).await;
  let identical = insert_fact(&pool, bank, Some(&[1.0, 0.0])).await;
  let diagonal = insert_fact(&pool, bank, Some(&[1.0, 1.0])).await;
  insert_fact(&pool, bank, Some(&[0.0, 1.0])).await;
  insert_fact(&pool, bank, None).await;
  insert_fact(&pool, BankId::new(), Some(&[1.0, 0.0])).await;

  let results = store
    .search_vector(bank, &[1.0, 0.0], 2)
    .await
    .expect("search");

  assert_eq!(results.len(), 2);
  assert_eq!(
    results[0].0, identical,
    "better candidates replace the early decoy"
  );
  assert!((results[0].1 - 1.0).abs() < 1e-6);
  assert_eq!(results[1].0, diagonal);
  assert!((results[1].1 - 0.707_106_77).abs() < 1e-6);
  assert!(
    !results.iter().any(|(id, _)| *id == far),
    "the decoy was evicted"
  );
}

#[tokio::test]
async fn equal_scores_keep_the_smaller_id() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();

  let first = insert_fact(&pool, bank, Some(&[1.0, 0.0])).await;
  let second = insert_fact(&pool, bank, Some(&[1.0, 0.0])).await;
  let expected = if first <= second { first } else { second };

  let results = store
    .search_vector(bank, &[1.0, 0.0], 1)
    .await
    .expect("search");

  assert_eq!(results.len(), 1);
  assert_eq!(results[0].0, expected, "ties are broken by the smaller id");
}

#[tokio::test]
async fn equal_scores_are_ordered_by_fact_id() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();

  let first = insert_fact(&pool, bank, Some(&[1.0, 0.0])).await;
  let second = insert_fact(&pool, bank, Some(&[1.0, 0.0])).await;

  let results = store
    .search_vector(bank, &[1.0, 0.0], 2)
    .await
    .expect("search");

  let ids: Vec<engine::types::FactId> = results.iter().map(|(id, _)| *id).collect();
  let mut expected = vec![first, second];
  expected.sort();
  assert_eq!(ids, expected);
}

#[tokio::test]
async fn rejects_a_dimension_mismatch() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();

  insert_fact(&pool, bank, Some(&[1.0, 0.0, 0.0])).await;

  let error = store
    .search_vector(bank, &[1.0, 0.0], 5)
    .await
    .expect_err("dimension mismatch");

  assert!(matches!(error, engine::Error::Config(_)), "error: {error}");
}

#[tokio::test]
async fn rejects_a_corrupt_embedding_blob() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();
  let id = engine::types::FactId::new();

  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, embedding, created_at, updated_at) \
     VALUES (?, ?, 'world', 'fact', ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(vec![1_u8, 2, 3])
  .execute(&pool)
  .await
  .expect("insert fact");

  let error = store
    .search_vector(bank, &[1.0], 5)
    .await
    .expect_err("corrupt blob");

  assert!(matches!(error, engine::Error::Fatal(_)), "error: {error}");
}

#[tokio::test]
async fn zero_k_and_empty_banks_return_empty() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();

  insert_fact(&pool, bank, Some(&[1.0, 0.0])).await;

  assert!(
    store
      .search_vector(bank, &[1.0, 0.0], 0)
      .await
      .expect("k=0")
      .is_empty()
  );
  assert!(
    store
      .search_vector(BankId::new(), &[1.0, 0.0], 5)
      .await
      .expect("empty bank")
      .is_empty()
  );
}
