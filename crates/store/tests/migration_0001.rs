//! Migration 0001: table inventory, constraints, and the FTS index.

use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;

const MIGRATION: &str = include_str!("../migrations/sqlite/0001_init.sql");

async fn pool() -> SqlitePool {
  let pool = SqlitePoolOptions::new()
    .max_connections(1)
    .connect("sqlite::memory:")
    .await
    .expect("open in-memory sqlite");
  sqlx::raw_sql(MIGRATION)
    .execute(&pool)
    .await
    .expect("apply migration 0001");
  pool
}

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

async fn insert_chunk(pool: &SqlitePool, id: &str, hash: &str) -> Result<(), sqlx::Error> {
  sqlx::query(
    "INSERT INTO chunk (id, bank_id, source_id, chunk_index, text, content_hash, status, \
     time_start, time_end, created_at, updated_at) \
     VALUES (?, 'b1', 's1', 0, 'hello world', ?, 'pending', NULL, NULL, 0, 0)",
  )
  .bind(id)
  .bind(hash)
  .execute(pool)
  .await
  .map(|_| ())
}

async fn insert_fact(
  pool: &SqlitePool,
  id: &str,
  fact_type: &str,
  chunk_id: Option<&str>,
  confidence: Option<f64>,
) -> Result<(), sqlx::Error> {
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, chunk_id, confidence, created_at, updated_at) \
     VALUES (?, 'b1', ?, 'a fact', ?, ?, 0, 0)",
  )
  .bind(id)
  .bind(fact_type)
  .bind(chunk_id)
  .bind(confidence)
  .execute(pool)
  .await
  .map(|_| ())
}

async fn insert_link(
  pool: &SqlitePool,
  from: &str,
  to: &str,
  link_type: &str,
  relation: Option<&str>,
  weight: f64,
) -> Result<(), sqlx::Error> {
  sqlx::query(
    "INSERT INTO fact_link (id, bank_id, from_fact_id, to_fact_id, link_type, relation, weight, \
     created_at, updated_at) VALUES (?, 'b1', ?, ?, ?, ?, ?, 0, 0)",
  )
  .bind(format!("l-{from}-{to}-{link_type}"))
  .bind(from)
  .bind(to)
  .bind(link_type)
  .bind(relation)
  .bind(weight)
  .execute(pool)
  .await
  .map(|_| ())
}

#[tokio::test]
async fn creates_seven_core_tables_and_the_fts_index() {
  let pool = pool().await;

  let tables: Vec<String> = sqlx::query_scalar(
    "SELECT name FROM sqlite_master WHERE type = 'table' \
     AND name IN ('bank','source','chunk','fact','entity','fact_entity','fact_link','fact_fts') \
     ORDER BY name",
  )
  .fetch_all(&pool)
  .await
  .expect("list tables");

  assert_eq!(
    tables,
    vec![
      "bank".to_string(),
      "chunk".to_string(),
      "entity".to_string(),
      "fact".to_string(),
      "fact_entity".to_string(),
      "fact_fts".to_string(),
      "fact_link".to_string(),
      "source".to_string(),
    ]
  );

  let indexes: Vec<String> = sqlx::query_scalar(
    "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%' ORDER BY name",
  )
  .fetch_all(&pool)
  .await
  .expect("list indexes");
  assert_eq!(indexes.len(), 8, "indexes: {indexes:?}");
}

#[tokio::test]
async fn rejects_duplicate_chunk_hash_within_a_bank() {
  let pool = pool().await;
  seed_bank(&pool).await;
  seed_source(&pool).await;

  insert_chunk(&pool, "c1", "hash-a")
    .await
    .expect("first chunk");
  let error = insert_chunk(&pool, "c2", "hash-a")
    .await
    .expect_err("duplicate hash");
  assert!(error.to_string().contains("UNIQUE"), "{error}");
}

#[tokio::test]
async fn rejects_duplicate_entity_canonical_name() {
  let pool = pool().await;
  seed_bank(&pool).await;

  let insert = |id: &'static str| {
    sqlx::query(
      "INSERT INTO entity (id, bank_id, canonical_name, entity_type, mention_count, \
       created_at, updated_at) VALUES (?, 'b1', 'emily', 'person', 0, 0, 0)",
    )
    .bind(id)
    .execute(&pool)
  };

  insert("e1").await.expect("first entity");
  let error = insert("e2").await.expect_err("duplicate name");
  assert!(error.to_string().contains("UNIQUE"), "{error}");
}

#[tokio::test]
async fn enforces_fact_conditional_constraints() {
  let pool = pool().await;
  seed_bank(&pool).await;

  // Reflect-produced opinion without a chunk must carry a confidence.
  insert_fact(&pool, "f1", "opinion", None, None)
    .await
    .expect_err("opinion without confidence");

  insert_fact(&pool, "f2", "opinion", None, Some(0.7))
    .await
    .expect("opinion with confidence");

  // Observations never come from a chunk.
  insert_fact(&pool, "f3", "observation", None, None)
    .await
    .expect("observation without chunk");
  insert_fact(&pool, "f4", "observation", Some("c1"), None)
    .await
    .expect_err("observation with chunk");

  // Enum domain and range checks.
  insert_fact(&pool, "f5", "nonsense", None, None)
    .await
    .expect_err("invalid fact type");
  insert_fact(&pool, "f6", "opinion", Some("c1"), Some(1.5))
    .await
    .expect_err("confidence out of range");
}

#[tokio::test]
async fn enforces_link_shape_and_weight() {
  let pool = pool().await;
  seed_bank(&pool).await;

  insert_link(&pool, "a", "b", "temporal", None, 0.5)
    .await
    .expect("temporal pair row");
  insert_link(&pool, "a", "b", "semantic", Some("causes"), 0.5)
    .await
    .expect_err("relation on non-causal edge");
  insert_link(&pool, "a", "b", "causal", None, 0.5)
    .await
    .expect_err("causal edge without relation");
  insert_link(&pool, "a", "b", "causal", Some("causes"), 1.0)
    .await
    .expect("causal edge");
  insert_link(&pool, "a", "c", "temporal", None, 1.5)
    .await
    .expect_err("weight out of range");
}

#[tokio::test]
async fn unique_edge_treats_null_relation_as_one_value() {
  let pool = pool().await;
  seed_bank(&pool).await;

  insert_link(&pool, "a", "b", "temporal", None, 0.5)
    .await
    .expect("first temporal edge");
  let error = insert_link(&pool, "a", "b", "temporal", None, 0.5)
    .await
    .expect_err("duplicate temporal edge");
  assert!(error.to_string().contains("UNIQUE"), "{error}");
}

#[tokio::test]
async fn fts_accepts_rows_and_answers_matches() {
  let pool = pool().await;

  for (fact_id, text) in [
    ("f1", "Emily works at Google as a software engineer."),
    ("f2", "The weather in Berlin was cold."),
  ] {
    sqlx::query("INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)")
      .bind(fact_id)
      .bind(text)
      .execute(&pool)
      .await
      .expect("insert fts row");
  }

  let hits: Vec<String> = sqlx::query_scalar(
    "SELECT fact_id FROM fact_fts WHERE fact_fts MATCH 'Emily' ORDER BY bm25(fact_fts) LIMIT 5",
  )
  .fetch_all(&pool)
  .await
  .expect("fts query");

  assert_eq!(hits, vec!["f1".to_string()]);
}

#[tokio::test]
async fn rejects_invalid_json_in_attrs() {
  let pool = pool().await;
  seed_bank(&pool).await;

  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, attrs, created_at, updated_at) \
     VALUES ('f1', 'b1', 'world', 'a fact', '{\"k\": 1}', 0, 0)",
  )
  .execute(&pool)
  .await
  .expect("valid json");

  let error = sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, attrs, created_at, updated_at) \
     VALUES ('f2', 'b1', 'world', 'a fact', 'not json', 0, 0)",
  )
  .execute(&pool)
  .await
  .expect_err("invalid json");
  assert!(error.to_string().contains("CHECK"), "{error}");
}
