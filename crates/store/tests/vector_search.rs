//! P1-13: vector search through the schema-managed sqlite-vec index.

use engine::ports::Store;
use engine::types::{BankId, FactId};
use sqlx::{Sqlite, SqlitePool, Transaction};
use store::testing::memory_pool;
use store::{EMBEDDING_DIM, SqliteStore};

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

async fn insert_fact(
  pool: &SqlitePool,
  bank: BankId,
  embedding: Option<&[f32]>,
) -> Result<FactId, sqlx::Error> {
  let id = FactId::new();

  // fact and fact_vec are 1:1; mirror tx2's single-transaction write.
  let mut tx = pool.begin().await?;
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, created_at, updated_at) \
     VALUES (?, ?, 'world', 'fact', 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .execute(&mut *tx)
  .await?;

  if let Some(embedding) = embedding {
    insert_embedding(&mut tx, id, embedding).await?;
  }

  tx.commit().await?;
  Ok(id)
}

async fn insert_embedding(
  tx: &mut Transaction<'_, Sqlite>,
  id: FactId,
  embedding: &[f32],
) -> Result<(), sqlx::Error> {
  sqlx::query(
    "INSERT INTO fact_vec(rowid, fact_id, bank_id, embedding) \
     SELECT rowid, id, bank_id, ? FROM fact WHERE id = ?",
  )
  .bind(blob(embedding))
  .bind(id.to_string())
  .execute(&mut **tx)
  .await
  .map(|_| ())
}

async fn replace_embedding(
  pool: &SqlitePool,
  id: FactId,
  embedding: &[f32],
) -> Result<(), sqlx::Error> {
  let mut tx = pool.begin().await?;
  sqlx::query("DELETE FROM fact_vec WHERE rowid = (SELECT rowid FROM fact WHERE id = ?)")
    .bind(id.to_string())
    .execute(&mut *tx)
    .await?;
  insert_embedding(&mut tx, id, embedding).await?;
  tx.commit().await
}

#[tokio::test]
async fn ranks_by_cosine_and_applies_the_limit() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();

  let far = vector(&[(0, -1.0)]);
  let identical = vector(&[(0, 1.0)]);
  let diagonal = vector(&[(0, 1.0), (1, 1.0)]);
  let orthogonal = vector(&[(1, 1.0)]);
  let query = vector(&[(0, 1.0)]);

  let far = insert_fact(&pool, bank, Some(&far)).await.expect("insert");
  let identical = insert_fact(&pool, bank, Some(&identical))
    .await
    .expect("insert");
  let diagonal = insert_fact(&pool, bank, Some(&diagonal))
    .await
    .expect("insert");
  insert_fact(&pool, bank, Some(&orthogonal))
    .await
    .expect("insert");
  insert_fact(&pool, bank, None)
    .await
    .expect("null embedding");
  insert_fact(&pool, BankId::new(), Some(&query))
    .await
    .expect("other bank");

  let results = store.search_vector(bank, &query, 2).await.expect("search");

  assert_eq!(results.len(), 2);
  assert_eq!(results[0].0, identical);
  assert!((results[0].1 - 1.0).abs() < 1e-6);
  assert_eq!(results[1].0, diagonal);
  assert!((results[1].1 - 0.707_106_77).abs() < 1e-5);
  assert!(
    !results.iter().any(|(id, _)| *id == far),
    "a negative-cosine candidate is below the top two"
  );
}

#[tokio::test]
async fn equal_scores_keep_the_smaller_id() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();
  let query = vector(&[(0, 1.0)]);

  let first = insert_fact(&pool, bank, Some(&query))
    .await
    .expect("insert");
  let second = insert_fact(&pool, bank, Some(&query))
    .await
    .expect("insert");
  let expected = if first <= second { first } else { second };

  let results = store.search_vector(bank, &query, 1).await.expect("search");

  assert_eq!(results.len(), 1);
  assert_eq!(results[0].0, expected, "ties are broken by the smaller id");
}

#[tokio::test]
async fn equal_scores_are_ordered_by_fact_id() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();
  let query = vector(&[(0, 1.0)]);

  let first = insert_fact(&pool, bank, Some(&query))
    .await
    .expect("insert");
  let second = insert_fact(&pool, bank, Some(&query))
    .await
    .expect("insert");

  let results = store.search_vector(bank, &query, 2).await.expect("search");

  let ids: Vec<FactId> = results.iter().map(|(id, _)| *id).collect();
  let mut expected = vec![first, second];
  expected.sort();
  assert_eq!(ids, expected);
}

#[tokio::test]
async fn rejects_a_query_with_the_wrong_dimension() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();

  insert_fact(&pool, bank, Some(&vector(&[(0, 1.0)])))
    .await
    .expect("insert");

  let error = store
    .search_vector(bank, &[1.0, 0.0], 5)
    .await
    .expect_err("dimension mismatch");

  assert!(matches!(error, engine::Error::Config(_)), "error: {error}");
}

#[tokio::test]
async fn rejects_a_wrong_dimension_at_write_time() {
  let pool = memory_pool().await.expect("pool");
  let bank = BankId::new();

  let error = insert_fact(&pool, bank, Some(&[1.0, 0.0]))
    .await
    .expect_err("the vector table enforces the schema dimension");
  assert!(!error.to_string().is_empty());

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(&pool)
    .await
    .expect("count facts");
  let vectors: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_vec")
    .fetch_one(&pool)
    .await
    .expect("count vectors");
  assert_eq!(facts, 0, "the transaction rolled the fact back");
  assert_eq!(vectors, 0, "the invalid vector was rejected");
}

#[tokio::test]
async fn rejects_a_corrupt_embedding_blob_at_write_time() {
  let pool = memory_pool().await.expect("pool");
  let bank = BankId::new();
  let id = FactId::new();

  let mut tx = pool.begin().await.expect("begin");
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, created_at, updated_at) \
     VALUES (?, ?, 'world', 'corrupt', 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .execute(&mut *tx)
  .await
  .expect("insert fact");

  let error = sqlx::query(
    "INSERT INTO fact_vec(rowid, fact_id, bank_id, embedding) \
     SELECT rowid, id, bank_id, ? FROM fact WHERE id = ?",
  )
  .bind(vec![1_u8, 2, 3])
  .bind(id.to_string())
  .execute(&mut *tx)
  .await
  .expect_err("a malformed blob cannot enter the vector table");

  assert!(!error.to_string().is_empty());
  tx.rollback().await.expect("rollback");

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(&pool)
    .await
    .expect("count facts");
  assert_eq!(facts, 0, "the transaction rolled the fact back");
}

#[tokio::test]
async fn zero_k_and_empty_banks_return_empty() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();
  let query = vector(&[(0, 1.0)]);

  insert_fact(&pool, bank, Some(&query))
    .await
    .expect("insert");

  assert!(
    store
      .search_vector(bank, &query, 0)
      .await
      .expect("k=0")
      .is_empty()
  );
  assert!(
    store
      .search_vector(BankId::new(), &query, 5)
      .await
      .expect("empty bank")
      .is_empty()
  );
}

#[tokio::test]
async fn fact_deletion_removes_its_vector_and_embeddings_can_be_replaced() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();
  let query = vector(&[(0, 1.0)]);

  let first = insert_fact(&pool, bank, Some(&query))
    .await
    .expect("insert");
  let results = store
    .search_vector(bank, &query, 1)
    .await
    .expect("search after insert");
  assert_eq!(results[0].0, first);

  let second = insert_fact(&pool, bank, Some(&query))
    .await
    .expect("insert");
  let results = store
    .search_vector(bank, &query, 2)
    .await
    .expect("search after second insert");
  let ids: Vec<FactId> = results.iter().map(|(id, _)| *id).collect();
  assert_eq!(ids, vec![first, second]);

  // Deleting the fact row removes its vector through the schema trigger.
  sqlx::query("DELETE FROM fact WHERE id = ?")
    .bind(first.to_string())
    .execute(&pool)
    .await
    .expect("delete fact");
  let results = store
    .search_vector(bank, &query, 2)
    .await
    .expect("search after delete");
  assert_eq!(results.len(), 1);
  assert_eq!(results[0].0, second);

  // Replacing an embedding deletes and re-inserts the vector row.
  let orthogonal = vector(&[(1, 1.0)]);
  replace_embedding(&pool, second, &orthogonal)
    .await
    .expect("replace embedding");
  let results = store
    .search_vector(bank, &query, 2)
    .await
    .expect("search after replacement");
  assert_eq!(results.len(), 1);
  assert_eq!(results[0].0, second);
  assert!(results[0].1.abs() < 1e-5, "score: {}", results[0].1);
}
