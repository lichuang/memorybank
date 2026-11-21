//! `increment_access`: batched hit counters on the fact rows.

use engine::ports::Store;
use engine::types::{BankId, FactId};
use store::SqliteStore;
use store::testing::memory_pool;

async fn insert_fact(store: &SqliteStore, bank: BankId) -> FactId {
  let id = FactId::new();
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, created_at, updated_at) \
     VALUES (?, ?, 'world', 'fact', 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .execute(store.pool())
  .await
  .expect("fact");
  id
}

#[tokio::test]
async fn increments_each_listed_fact_once_per_call() {
  let store = SqliteStore::new(memory_pool().await.expect("pool"));
  let bank = BankId::new();
  let first = insert_fact(&store, bank).await;
  let second = insert_fact(&store, bank).await;
  let untouched = insert_fact(&store, bank).await;

  store
    .increment_access(&[first, second, first])
    .await
    .expect("increment");

  assert_eq!(
    access_count(&store, first).await,
    1,
    "duplicates within a call count once"
  );
  assert_eq!(access_count(&store, second).await, 1);
  assert_eq!(access_count(&store, untouched).await, 0);

  store.increment_access(&[first]).await.expect("second call");
  assert_eq!(
    access_count(&store, first).await,
    2,
    "the counter accumulates across calls"
  );
}

async fn access_count(store: &SqliteStore, id: FactId) -> i64 {
  sqlx::query_scalar::<_, i64>("SELECT access_count FROM fact WHERE id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("count")
}

#[tokio::test]
async fn empty_input_is_a_no_op_and_updated_at_is_untouched() {
  let store = SqliteStore::new(memory_pool().await.expect("pool"));
  let bank = BankId::new();
  let id = insert_fact(&store, bank).await;

  store.increment_access(&[]).await.expect("empty");

  let (before, updated_at): (i64, i64) =
    sqlx::query_as("SELECT access_count, updated_at FROM fact WHERE id = ?")
      .bind(id.to_string())
      .fetch_one(store.pool())
      .await
      .expect("row");
  assert_eq!(before, 0);

  store.increment_access(&[id]).await.expect("increment");

  let (after, later): (i64, i64) =
    sqlx::query_as("SELECT access_count, updated_at FROM fact WHERE id = ?")
      .bind(id.to_string())
      .fetch_one(store.pool())
      .await
      .expect("row");
  assert_eq!(after, 1);
  assert_eq!(later, updated_at, "recall hits do not touch updated_at");
}

#[tokio::test]
async fn unknown_ids_are_ignored() {
  let store = SqliteStore::new(memory_pool().await.expect("pool"));
  let bank = BankId::new();
  let id = insert_fact(&store, bank).await;

  store
    .increment_access(&[id, FactId::new(), FactId::new()])
    .await
    .expect("unknown ids");

  let count: i64 = sqlx::query_scalar("SELECT access_count FROM fact WHERE id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("count");
  assert_eq!(count, 1);
}

#[tokio::test]
async fn large_batches_are_chunked() {
  let store = SqliteStore::new(memory_pool().await.expect("pool"));
  let bank = BankId::new();
  let mut ids = Vec::new();
  for _ in 0..1_001 {
    ids.push(insert_fact(&store, bank).await);
  }

  store.increment_access(&ids).await.expect("batch");

  let total: i64 = sqlx::query_scalar("SELECT SUM(access_count) FROM fact")
    .fetch_one(store.pool())
    .await
    .expect("sum");
  assert_eq!(total, 1_001);
}
