//! P1-13: sqlite-vec registration and the index created by migration 0001.

use store::testing::memory_pool;

#[tokio::test]
async fn vec0_and_its_triggers_are_part_of_the_initial_schema() {
  let pool = memory_pool().await.expect("pool");

  let version: String = sqlx::query_scalar("SELECT vec_version()")
    .fetch_one(&pool)
    .await
    .expect("vec_version");
  assert!(version.starts_with('v'), "version: {version}");

  let table: Option<String> =
    sqlx::query_scalar("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'fact_vec'")
      .fetch_optional(&pool)
      .await
      .expect("table lookup");
  assert_eq!(table.as_deref(), Some("fact_vec"));

  let triggers: Vec<String> = sqlx::query_scalar(
    "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'fact_vec_%' \
     ORDER BY name",
  )
  .fetch_all(&pool)
  .await
  .expect("trigger lookup");
  assert_eq!(triggers, vec!["fact_vec_delete"]);
}
