//! P1-12 support: occurrence-interval and fact read queries.

use engine::ports::Store;
use engine::testing::timestamp;
use engine::types::{
  BankId, ClosedInterval, FactId, OccurrenceInterval, TimeRange, to_epoch_millis,
};
use sqlx::SqlitePool;
use store::SqliteStore;
use store::testing::memory_pool;

fn ts(millis: i64) -> engine::types::Timestamp {
  timestamp(millis)
}

async fn insert_interval(
  pool: &SqlitePool,
  bank: BankId,
  start: Option<i64>,
  end: Option<i64>,
) -> FactId {
  let id = FactId::new();
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, occurred_start, occurred_end, \
     created_at, updated_at) VALUES (?, ?, 'world', 'fact', ?, ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(start)
  .bind(end)
  .execute(pool)
  .await
  .expect("insert fact");

  id
}

fn hour(value: i64) -> i64 {
  value * 60 * 60 * 1_000
}

#[tokio::test]
async fn facts_in_range_matches_occurrence_intersections() {
  let pool = memory_pool().await.expect("pool");
  let bank = BankId::new();
  let other_bank = BankId::new();

  let before = insert_interval(&pool, bank, Some(-hour(2)), Some(-hour(1))).await;
  let overlap = insert_interval(&pool, bank, Some(-hour(1) / 2), Some(hour(1) / 2)).await;
  let inside = insert_interval(&pool, bank, Some(0), Some(hour(1) / 6)).await;
  let after = insert_interval(&pool, bank, Some(hour(30)), Some(hour(31))).await;
  let open = insert_interval(&pool, bank, Some(hour(2)), None).await;
  insert_interval(&pool, bank, None, None).await;
  insert_interval(&pool, other_bank, Some(0), Some(0)).await;

  let store = SqliteStore::new(pool.clone());

  let ids = store
    .facts_in_range(
      bank,
      TimeRange {
        start: ts(-hour(1)),
        end: ts(hour(1)),
      },
    )
    .await
    .expect("range");
  assert_eq!(
    ids,
    vec![before, overlap, inside],
    "closed intervals: a fact ending exactly at range.start still intersects"
  );

  let ids = store
    .facts_in_range(
      bank,
      TimeRange {
        start: ts(hour(2)),
        end: ts(hour(3)),
      },
    )
    .await
    .expect("open end");
  assert_eq!(ids, vec![open], "a NULL end means the interval stays open");

  let ids = store
    .facts_in_range(
      bank,
      TimeRange {
        start: ts(0),
        end: ts(0),
      },
    )
    .await
    .expect("point range");
  assert_eq!(ids, vec![overlap, inside]);
  assert!(
    !ids.contains(&after),
    "facts outside the range are never returned"
  );
}

fn blob(values: &[f32]) -> Vec<u8> {
  values
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect()
}

#[tokio::test]
async fn fetch_facts_round_trips_every_field() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();
  let id = FactId::new();

  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, embedding, chunk_id, occurred_start, \
     occurred_end, mentioned_at, confidence, access_count, attrs, created_at, updated_at) \
     VALUES (?, ?, 'world', 'a fact', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(blob(&[1.5, -2.0]))
  .bind(hour(1))
  .bind(hour(2))
  .bind(hour(3))
  .bind(0.75_f64)
  .bind(3_i64)
  .bind(r#"{"fact_kind":"statement"}"#)
  .bind(1_000_i64)
  .bind(2_000_i64)
  .execute(&pool)
  .await
  .expect("insert fact");

  let facts = store.fetch_facts(&[id]).await.expect("fetch");
  assert_eq!(facts.len(), 1);
  let fact = &facts[0];

  assert_eq!(fact.id, id);
  assert_eq!(fact.bank_id, bank);
  assert_eq!(fact.text, "a fact");
  assert_eq!(fact.embedding.as_deref(), Some(&[1.5_f32, -2.0][..]));
  assert_eq!(fact.chunk_id, None);
  assert_eq!(
    fact.occurred,
    OccurrenceInterval::Closed(
      ClosedInterval::new(ts(hour(1)), ts(hour(2))).expect("ordered interval")
    )
  );
  assert_eq!(fact.mentioned_at, Some(ts(hour(3))));
  assert_eq!(fact.confidence, Some(0.75));
  assert_eq!(fact.access_count, 3);
  assert_eq!(
    fact.attrs,
    Some(serde_json::json!({ "fact_kind": "statement" }))
  );
  assert_eq!(to_epoch_millis(fact.created_at), 1_000);
  assert_eq!(to_epoch_millis(fact.updated_at), 2_000);
}

#[tokio::test]
async fn fetch_facts_skips_unknown_and_empty_requests() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();

  let known = insert_interval(&pool, bank, Some(0), Some(0)).await;

  let facts = store
    .fetch_facts(&[known, FactId::new()])
    .await
    .expect("fetch");
  assert_eq!(facts.len(), 1);
  assert_eq!(facts[0].id, known);
  assert_eq!(facts[0].embedding, None);

  assert!(
    store.fetch_facts(&[]).await.expect("empty").is_empty(),
    "an empty request does not hit the database"
  );
}

#[tokio::test]
async fn fetch_facts_rejects_a_corrupt_embedding_blob() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();
  let id = FactId::new();

  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, embedding, created_at, updated_at) \
     VALUES (?, ?, 'world', 'corrupt', ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(vec![1_u8, 2, 3])
  .execute(&pool)
  .await
  .expect("insert fact");

  let error = store.fetch_facts(&[id]).await.expect_err("corrupt blob");

  assert!(matches!(error, engine::Error::Fatal(_)), "error: {error}");
}

#[tokio::test]
async fn skips_rows_with_invalid_occurrence_intervals() {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = BankId::new();

  let valid = insert_interval(&pool, bank, Some(0), Some(hour(1))).await;
  let reversed = insert_interval(&pool, bank, Some(hour(2)), Some(hour(1))).await;
  let end_only = insert_interval(&pool, bank, None, Some(hour(1))).await;

  let facts = store
    .fetch_facts(&[valid, reversed, end_only])
    .await
    .expect("fetch");
  assert_eq!(
    facts.len(),
    1,
    "invalid rows fail individually and do not abort the read"
  );
  assert_eq!(facts[0].id, valid);

  let ids = store
    .facts_in_range(
      bank,
      TimeRange {
        start: ts(-hour(1)),
        end: ts(hour(3)),
      },
    )
    .await
    .expect("range");
  assert_eq!(ids, vec![valid], "invalid rows are filtered in SQL");
}
