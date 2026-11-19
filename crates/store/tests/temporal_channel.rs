//! Temporal recall channel over occurrence intervals.

use engine::ports::{FactScope, Store, TemporalParser};
use engine::recall::{TemporalConfig, temporal_channel};
use engine::testing::{RuleTemporalParser, timestamp};
use engine::types::{Bank, BankId, BankProfile, FactId, FactType, TimeRange};
use sqlx::SqlitePool;
use store::SqliteStore;
use store::testing::memory_pool;

fn ts(millis: i64) -> engine::types::Timestamp {
  timestamp(millis)
}

fn hour(value: i64) -> i64 {
  value * 60 * 60 * 1_000
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

async fn insert_fact(
  pool: &SqlitePool,
  bank: BankId,
  fact_type: FactType,
  start: Option<i64>,
  end: Option<i64>,
) -> FactId {
  let id = FactId::new();
  let confidence = (fact_type == FactType::Opinion).then_some(0.5_f64);
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, occurred_start, occurred_end, \
     confidence, created_at, updated_at) VALUES (?, ?, ?, 'fact', ?, ?, ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(fact_type.as_str())
  .bind(start)
  .bind(end)
  .bind(confidence)
  .execute(pool)
  .await
  .expect("fact");
  id
}

struct FixedRangeParser(TimeRange);

impl TemporalParser for FixedRangeParser {
  fn parse(&self, _query: &str, _now: engine::types::Timestamp) -> Option<TimeRange> {
    Some(self.0)
  }
}

#[tokio::test]
async fn scores_by_proximity_to_the_query_midpoint() {
  let (store, bank) = setup().await;
  // Query range: [0, 10h]; midpoint 5h, half width 5h.
  let range = TimeRange {
    start: ts(0),
    end: ts(hour(10)),
  };
  let parser = FixedRangeParser(range);

  let center = insert_fact(
    store.pool(),
    bank,
    FactType::World,
    Some(hour(5)),
    Some(hour(5)),
  )
  .await;
  let near = insert_fact(
    store.pool(),
    bank,
    FactType::World,
    Some(hour(6)),
    Some(hour(6)),
  )
  .await;
  let edge = insert_fact(
    store.pool(),
    bank,
    FactType::World,
    Some(hour(10)),
    Some(hour(10)),
  )
  .await;
  insert_fact(store.pool(), bank, FactType::World, None, None).await;

  let config = TemporalConfig {
    top_k: 10,
    min_score: 0.0,
  };
  let hits = temporal_channel(
    &store,
    bank,
    "irrelevant",
    ts(0),
    &parser,
    &config,
    &FactScope::All,
  )
  .await
  .expect("temporal");

  assert_eq!(
    hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![center, near, edge],
    "closest first; unknown occurrences are dropped"
  );
  assert!((hits[0].1 - 1.0).abs() < 1e-6);
  assert!((hits[1].1 - 0.8).abs() < 1e-6);
  assert!((hits[2].1 - 0.0).abs() < 1e-6);
}

#[tokio::test]
async fn applies_min_score_and_top_k() {
  let (store, bank) = setup().await;
  let range = TimeRange {
    start: ts(0),
    end: ts(hour(10)),
  };
  let parser = FixedRangeParser(range);
  let center = insert_fact(
    store.pool(),
    bank,
    FactType::World,
    Some(hour(5)),
    Some(hour(5)),
  )
  .await;
  insert_fact(
    store.pool(),
    bank,
    FactType::World,
    Some(hour(9)),
    Some(hour(9)),
  )
  .await;

  let filtered = TemporalConfig {
    top_k: 10,
    min_score: 0.5,
  };
  let hits = temporal_channel(
    &store,
    bank,
    "q",
    ts(0),
    &parser,
    &filtered,
    &FactScope::All,
  )
  .await
  .expect("filtered");
  assert_eq!(
    hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![center]
  );

  let limited = TemporalConfig {
    top_k: 1,
    min_score: 0.0,
  };
  let hits = temporal_channel(&store, bank, "q", ts(0), &parser, &limited, &FactScope::All)
    .await
    .expect("limited");
  assert_eq!(hits.len(), 1);
  assert_eq!(hits[0].0, center);
}

#[tokio::test]
async fn rule_parser_drives_the_channel() {
  let (store, bank) = setup().await;
  const DAY: i64 = 24 * 60 * 60 * 1_000;
  let now = ts(1_700_000_000_000);
  let today_start = (now.timestamp_millis() / DAY) * DAY;
  let yesterday_start = today_start - DAY;

  let inside = insert_fact(
    store.pool(),
    bank,
    FactType::World,
    Some(yesterday_start + hour(12)),
    Some(yesterday_start + hour(12)),
  )
  .await;
  insert_fact(
    store.pool(),
    bank,
    FactType::World,
    Some(today_start + hour(2)),
    Some(today_start + hour(2)),
  )
  .await;

  let config = TemporalConfig::default();
  let hits = temporal_channel(
    &store,
    bank,
    "what happened yesterday?",
    now,
    &RuleTemporalParser,
    &config,
    &FactScope::All,
  )
  .await
  .expect("rule parser");

  assert_eq!(
    hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![inside]
  );
}

#[tokio::test]
async fn no_time_expression_means_no_activation() {
  let (store, bank) = setup().await;
  insert_fact(store.pool(), bank, FactType::World, Some(0), Some(hour(1))).await;

  let config = TemporalConfig::default();
  let hits = temporal_channel(
    &store,
    bank,
    "what is Emily's job?",
    ts(0),
    &RuleTemporalParser,
    &config,
    &FactScope::All,
  )
  .await
  .expect("no time");
  assert!(hits.is_empty());
}

#[tokio::test]
async fn open_intervals_and_scope_are_handled() {
  let (store, bank) = setup().await;
  let range = TimeRange {
    start: ts(0),
    end: ts(hour(10)),
  };
  let parser = FixedRangeParser(range);

  let open = insert_fact(store.pool(), bank, FactType::World, Some(hour(5)), None).await;
  let opinion = insert_fact(
    store.pool(),
    bank,
    FactType::Opinion,
    Some(hour(5)),
    Some(hour(5)),
  )
  .await;

  let config = TemporalConfig::default();
  let all = temporal_channel(&store, bank, "q", ts(0), &parser, &config, &FactScope::All)
    .await
    .expect("all");
  assert!(
    all.iter().any(|(id, _)| *id == open),
    "open intervals score by start"
  );
  assert!(all.iter().any(|(id, _)| *id == opinion));

  let worlds = temporal_channel(
    &store,
    bank,
    "q",
    ts(0),
    &parser,
    &config,
    &FactScope::Types(vec![FactType::World]),
  )
  .await
  .expect("worlds");
  assert_eq!(
    worlds.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    vec![open]
  );

  let none = temporal_channel(
    &store,
    bank,
    "q",
    ts(0),
    &parser,
    &config,
    &FactScope::Types(Vec::new()),
  )
  .await
  .expect("none");
  assert!(none.is_empty());
}

#[tokio::test]
async fn zero_width_range_scores_are_finite() {
  let (store, bank) = setup().await;
  let instant = ts(hour(3));
  let parser = FixedRangeParser(TimeRange {
    start: instant,
    end: instant,
  });
  let exact = insert_fact(
    store.pool(),
    bank,
    FactType::World,
    Some(hour(3)),
    Some(hour(3)),
  )
  .await;
  insert_fact(
    store.pool(),
    bank,
    FactType::World,
    Some(hour(5)),
    Some(hour(5)),
  )
  .await;

  let config = TemporalConfig::default();
  let hits = temporal_channel(&store, bank, "q", ts(0), &parser, &config, &FactScope::All)
    .await
    .expect("instant");
  assert!(hits.iter().all(|(_, score)| score.is_finite()));
  assert!(hits.iter().any(|(id, _)| *id == exact));
}

#[tokio::test]
async fn disabled_channel_returns_empty() {
  let (store, bank) = setup().await;
  insert_fact(store.pool(), bank, FactType::World, Some(0), Some(hour(1))).await;

  let config = TemporalConfig {
    top_k: 0,
    ..TemporalConfig::default()
  };
  let hits = temporal_channel(
    &store,
    bank,
    "yesterday",
    ts(0),
    &RuleTemporalParser,
    &config,
    &FactScope::All,
  )
  .await
  .expect("disabled");
  assert!(hits.is_empty());
}
