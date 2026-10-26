//! Migration runner semantics: idempotency, isolation, and pooling.

#[tokio::test]
async fn migrate_is_idempotent_and_records_the_version() {
  let pool = store::testing::memory_pool().await.expect("open pool");

  // Second run must be a no-op, not an error.
  store::migrate(&pool).await.expect("re-run migrations");
  store::migrate(&pool).await.expect("third run");

  let applied: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM _sqlx_migrations")
    .fetch_one(&pool)
    .await
    .expect("count applied migrations");
  assert_eq!(applied, 1, "one embedded migration file");

  let succeeded: i64 =
    sqlx::query_scalar("SELECT COUNT(*) FROM _sqlx_migrations WHERE success = 1")
      .fetch_one(&pool)
      .await
      .expect("count successful migrations");
  assert_eq!(succeeded, 1);

  // The schema is usable after the runner.
  let tables: i64 = sqlx::query_scalar(
    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN \
     ('bank','source','chunk','fact','entity','fact_entity','fact_link','fact_fts')",
  )
  .fetch_one(&pool)
  .await
  .expect("count core tables");
  assert_eq!(tables, 8);
}

#[tokio::test]
async fn memory_pools_are_isolated() {
  let first = store::testing::memory_pool().await.expect("first pool");
  let second = store::testing::memory_pool().await.expect("second pool");

  sqlx::query(
    "INSERT INTO bank (id, name, background, skepticism, literalism, empathy, bias_strength, \
     created_at, updated_at) VALUES ('b1', 'bank', '', 3, 3, 3, 0.5, 0, 0)",
  )
  .execute(&first)
  .await
  .expect("insert into first");

  let first_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM bank")
    .fetch_one(&first)
    .await
    .expect("count first");
  let second_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM bank")
    .fetch_one(&second)
    .await
    .expect("count second");

  assert_eq!(first_count, 1);
  assert_eq!(second_count, 0, "each test database is isolated");
}

#[tokio::test]
async fn memory_pool_serves_concurrent_callers() {
  let pool = store::testing::memory_pool().await.expect("open pool");

  let (a, b, c, d) = tokio::join!(
    sqlx::query_scalar::<_, i64>("SELECT 1").fetch_one(&pool),
    sqlx::query_scalar::<_, i64>("SELECT 2").fetch_one(&pool),
    sqlx::query_scalar::<_, i64>("SELECT 3").fetch_one(&pool),
    sqlx::query_scalar::<_, i64>("SELECT 4").fetch_one(&pool),
  );

  assert_eq!(a.expect("a"), 1);
  assert_eq!(b.expect("b"), 2);
  assert_eq!(c.expect("c"), 3);
  assert_eq!(d.expect("d"), 4);
}

#[tokio::test]
async fn migrate_rejects_a_tampered_migration() {
  // A fresh database accepts the embedded migration; a recorded checksum that
  // no longer matches must be reported as a fatal error instead of re-running.
  let pool = store::testing::memory_pool().await.expect("open pool");

  sqlx::query("UPDATE _sqlx_migrations SET checksum = x'00' WHERE version = 1")
    .execute(&pool)
    .await
    .expect("tamper checksum");

  let error = store::migrate(&pool).await.expect_err("checksum mismatch");
  assert!(!error.is_retryable());
  assert!(
    matches!(error, engine::Error::Fatal(_)),
    "expected fatal, got {error}"
  );
}
