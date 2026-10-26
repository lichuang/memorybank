//! P0 smoke test: the migrated schema supports the inbox flow end to end
//! (bank → source → chunks → claim → release) plus the single-chunk transaction
//! boundary.
//!
//! P0 has no `Store` implementation yet (that lands with P1-3/P1-4), so this
//! exercises the same operations with runtime SQL. It de-risks the adapter work:
//! atomic claim, stale reclaim, release, and rollback semantics are proven
//! against the real schema.

use sqlx::SqlitePool;
use store::testing::memory_pool;

async fn seed_bank(pool: &SqlitePool) {
  sqlx::query(
    "INSERT INTO bank (id, name, background, skepticism, literalism, empathy, bias_strength, \
     created_at, updated_at) VALUES ('b1', 'bank', 'background', 3, 3, 3, 0.5, 0, 0)",
  )
  .execute(pool)
  .await
  .expect("insert bank");
}

async fn seed_source(pool: &SqlitePool) {
  sqlx::query(
    "INSERT INTO source (id, bank_id, title, payload_text, status, created_at, updated_at) \
     VALUES ('s1', 'b1', NULL, 'hello world', 'received', 0, 0)",
  )
  .execute(pool)
  .await
  .expect("insert source");
}

async fn seed_chunk(pool: &SqlitePool, id: &str, hash: &str, created_at: i64) {
  sqlx::query(
    "INSERT INTO chunk (id, bank_id, source_id, chunk_index, text, content_hash, status, \
     created_at, updated_at) VALUES (?, 'b1', 's1', 0, 'hello world', ?, 'pending', ?, ?)",
  )
  .bind(id)
  .bind(hash)
  .bind(created_at)
  .bind(created_at)
  .execute(pool)
  .await
  .expect("insert chunk");
}

/// Atomic claim of the oldest claimable chunk: pending, or `extracting` but
/// older than `stale_before`.
async fn claim(pool: &SqlitePool, now: i64, stale_before: i64) -> Option<String> {
  sqlx::query_scalar::<_, String>(
    "UPDATE chunk SET status = 'extracting', updated_at = ? \
     WHERE id = ( \
       SELECT id FROM chunk \
       WHERE bank_id = 'b1' \
         AND (status = 'pending' OR (status = 'extracting' AND updated_at < ?)) \
       ORDER BY created_at, id LIMIT 1 \
     ) \
     RETURNING id",
  )
  .bind(now)
  .bind(stale_before)
  .fetch_optional(pool)
  .await
  .expect("claim chunk")
}

async fn status_of(pool: &SqlitePool, id: &str) -> (String, i64) {
  sqlx::query_as::<_, (String, i64)>("SELECT status, updated_at FROM chunk WHERE id = ?")
    .bind(id)
    .fetch_one(pool)
    .await
    .expect("chunk status")
}

#[tokio::test]
async fn inbox_flow_reaches_claim_and_release() {
  let pool = memory_pool().await.expect("open pool");
  seed_bank(&pool).await;
  seed_source(&pool).await;
  seed_chunk(&pool, "c1", "hash-1", 10).await;
  seed_chunk(&pool, "c2", "hash-2", 20).await;

  // tx1 advances the source once chunk rows exist.
  sqlx::query("UPDATE source SET status = 'chunked', updated_at = 5 WHERE id = 's1'")
    .execute(&pool)
    .await
    .expect("advance source");

  let source: (String, String) =
    sqlx::query_as("SELECT status, payload_text FROM source WHERE id = 's1'")
      .fetch_one(&pool)
      .await
      .expect("source row");
  assert_eq!(source.0, "chunked");
  assert_eq!(source.1, "hello world");

  // Two claims hand out distinct chunks; the third finds nothing.
  assert_eq!(claim(&pool, 100, 50).await.as_deref(), Some("c1"));
  assert_eq!(
    status_of(&pool, "c1").await,
    ("extracting".to_string(), 100)
  );
  assert_eq!(claim(&pool, 100, 50).await.as_deref(), Some("c2"));
  assert_eq!(claim(&pool, 100, 50).await, None);

  // Release returns the chunk to pending, so it is claimable again.
  sqlx::query("UPDATE chunk SET status = 'pending', updated_at = 101 WHERE id = 'c1'")
    .execute(&pool)
    .await
    .expect("release chunk");
  assert_eq!(status_of(&pool, "c1").await, ("pending".to_string(), 101));
  assert_eq!(claim(&pool, 102, 50).await.as_deref(), Some("c1"));
}

#[tokio::test]
async fn stale_claims_are_reclaimed() {
  let pool = memory_pool().await.expect("open pool");
  seed_bank(&pool).await;
  seed_source(&pool).await;
  seed_chunk(&pool, "c1", "hash-1", 10).await;

  // First claim: claimed at t=100.
  assert_eq!(claim(&pool, 100, 50).await.as_deref(), Some("c1"));

  // While fresh (updated_at 100 is not < stale_before 100) the claim is not stolen.
  assert_eq!(claim(&pool, 110, 100).await, None);

  // Once stale (100 < 115), the same row is reclaimed and refreshed.
  assert_eq!(claim(&pool, 110, 115).await.as_deref(), Some("c1"));
  assert_eq!(
    status_of(&pool, "c1").await,
    ("extracting".to_string(), 110)
  );
}

#[tokio::test]
async fn failed_chunk_transaction_leaves_no_products() {
  let pool = memory_pool().await.expect("open pool");
  seed_bank(&pool).await;
  seed_source(&pool).await;
  seed_chunk(&pool, "c1", "hash-1", 10).await;

  let mut tx = pool.begin().await.expect("begin");

  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, chunk_id, created_at, updated_at) \
     VALUES ('f1', 'b1', 'world', 'a fact', 'c1', 20, 20)",
  )
  .execute(&mut *tx)
  .await
  .expect("insert fact");
  sqlx::query("UPDATE chunk SET status = 'done', updated_at = 20 WHERE id = 'c1'")
    .execute(&mut *tx)
    .await
    .expect("flip chunk");

  // Simulate a failure before commit.
  tx.rollback().await.expect("rollback");

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(&pool)
    .await
    .expect("count facts");
  assert_eq!(facts, 0, "no partial products");
  assert_eq!(status_of(&pool, "c1").await.0, "pending");
}
