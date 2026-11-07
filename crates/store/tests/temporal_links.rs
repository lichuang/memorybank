//! Temporal link construction against a real store.

use std::time::Duration;

use engine::link::{LinkConfig, build_temporal_links};
use engine::ports::{NewLink, Store};
use engine::testing::timestamp;
use engine::types::{
  Bank, BankId, BankProfile, Fact, FactId, FactType, LinkType, OccurrenceInterval, to_epoch_millis,
};
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

fn fact(bank: BankId, start: Option<i64>, end: Option<i64>) -> Fact {
  Fact {
    id: FactId::new(),
    bank_id: bank,
    fact_type: FactType::World,
    text: "fact".into(),
    embedding: None,
    chunk_id: None,
    occurred: OccurrenceInterval::from_parts(start.map(ts), end.map(ts))
      .expect("test interval must be valid"),
    mentioned_at: None,
    confidence: None,
    access_count: 0,
    attrs: None,
    created_at: ts(0),
    updated_at: ts(0),
  }
}

fn point(bank: BankId, at: i64) -> Fact {
  fact(bank, Some(at), Some(at))
}

async fn insert_existing(pool: &SqlitePool, fact: &Fact) {
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, occurred_start, occurred_end, \
     mentioned_at, created_at, updated_at) VALUES (?, ?, 'world', ?, ?, ?, ?, ?, ?)",
  )
  .bind(fact.id.to_string())
  .bind(fact.bank_id.to_string())
  .bind(&fact.text)
  .bind(fact.occurred.start().map(to_epoch_millis))
  .bind(fact.occurred.end().map(to_epoch_millis))
  .bind(fact.mentioned_at.map(to_epoch_millis))
  .bind(to_epoch_millis(fact.created_at))
  .bind(to_epoch_millis(fact.updated_at))
  .execute(pool)
  .await
  .expect("insert fact");
}

fn pairs(links: &[NewLink]) -> Vec<(FactId, FactId)> {
  let mut pairs: Vec<(FactId, FactId)> = links.iter().map(|link| (link.from, link.to)).collect();
  pairs.sort();
  pairs
}

fn config(window_hours: i64, sigma_seconds: u64, top_k: usize) -> LinkConfig {
  LinkConfig {
    temporal_window: Duration::from_secs(window_hours as u64 * 3_600),
    temporal_sigma: Duration::from_secs(sigma_seconds),
    temporal_top_k: top_k,
    ..LinkConfig::default()
  }
}

#[tokio::test]
async fn links_only_pairs_within_the_window() {
  let (store, bank) = setup().await;
  let a = fact(bank, Some(0), Some(hour(1)));
  let b = fact(bank, Some(hour(2)), Some(hour(3)));
  let c = fact(bank, Some(hour(30)), Some(hour(31)));
  let now = ts(9_999);

  let links = build_temporal_links(
    &store,
    bank,
    &[a.clone(), b.clone(), c.clone()],
    &config(4, 3_600, 5),
    now,
  )
  .await
  .expect("build");

  assert_eq!(links.len(), 2, "only A-B is inside the window");
  assert!(links.iter().all(|link| {
    link.link_type == LinkType::Temporal
      && link.relation.is_none()
      && link.created_at == now
      && (link.weight - (-1.0_f32).exp()).abs() < 1e-6
  }));

  let mut expected = vec![(a.id, b.id), (b.id, a.id)];
  expected.sort();
  assert_eq!(pairs(&links), expected);
}

#[tokio::test]
async fn per_fact_budget_keeps_the_closest_existing_facts() {
  let (store, bank) = setup().await;
  let b = point(bank, hour(1));
  let c = point(bank, hour(2));
  let d = point(bank, hour(3));
  for existing in [&b, &c, &d] {
    insert_existing(store.pool(), existing).await;
  }
  let a = point(bank, 0);

  let links = build_temporal_links(
    &store,
    bank,
    std::slice::from_ref(&a),
    &config(10, 3_600, 2),
    ts(1),
  )
  .await
  .expect("build");

  let mut expected = vec![(a.id, b.id), (b.id, a.id), (a.id, c.id), (c.id, a.id)];
  expected.sort();
  assert_eq!(pairs(&links), expected, "the budget keeps only B and C");
}

#[tokio::test]
async fn a_new_new_pair_is_written_once_as_reciprocal_rows() {
  let (store, bank) = setup().await;
  let a = point(bank, 0);
  let b = point(bank, hour(1));

  let links = build_temporal_links(
    &store,
    bank,
    &[a.clone(), b.clone()],
    &config(4, 3_600, 1),
    ts(1),
  )
  .await
  .expect("build");

  let mut expected = vec![(a.id, b.id), (b.id, a.id)];
  expected.sort();
  assert_eq!(
    pairs(&links),
    expected,
    "both sides select the pair, but it is written once"
  );
}

#[tokio::test]
async fn incomplete_time_facts_create_no_edges() {
  let (store, bank) = setup().await;

  let open = fact(bank, Some(0), None);
  let links = build_temporal_links(
    &store,
    bank,
    std::slice::from_ref(&open),
    &config(4, 3_600, 5),
    ts(1),
  )
  .await
  .expect("build");
  assert!(links.is_empty(), "the new fact has no complete interval");

  let open_existing = fact(bank, Some(0), None);
  insert_existing(store.pool(), &open_existing).await;
  let complete = point(bank, 0);
  let links = build_temporal_links(&store, bank, &[complete], &config(4, 3_600, 5), ts(1))
    .await
    .expect("build");
  assert!(
    links.is_empty(),
    "an existing open interval cannot be linked"
  );
}

#[tokio::test]
async fn the_window_boundary_is_inclusive() {
  let (store, bank) = setup().await;
  let edge = point(bank, hour(4));
  let beyond = point(bank, hour(4) + 1);
  insert_existing(store.pool(), &edge).await;
  insert_existing(store.pool(), &beyond).await;
  let new = point(bank, 0);

  let links = build_temporal_links(
    &store,
    bank,
    std::slice::from_ref(&new),
    &config(4, 3_600, 5),
    ts(1),
  )
  .await
  .expect("build");

  let mut expected = vec![(new.id, edge.id), (edge.id, new.id)];
  expected.sort();
  assert_eq!(pairs(&links), expected, "gap == window is still linked");
  assert!(
    (links[0].weight - (-4.0_f32).exp()).abs() < 1e-6,
    "weight: {}",
    links[0].weight
  );
}

#[tokio::test]
async fn existing_facts_from_another_bank_are_ignored() {
  let (store, bank) = setup().await;
  let foreign = point(BankId::new(), 0);
  insert_existing(store.pool(), &foreign).await;
  let new = point(bank, 0);

  let links = build_temporal_links(&store, bank, &[new], &config(4, 3_600, 5), ts(1))
    .await
    .expect("build");

  assert!(links.is_empty());
}

#[tokio::test]
async fn link_selection_is_deterministic() {
  let (store, bank) = setup().await;
  let existing = point(bank, hour(2));
  insert_existing(store.pool(), &existing).await;
  let a = fact(bank, Some(0), Some(hour(1)));
  let b = point(bank, hour(1));

  let first = build_temporal_links(
    &store,
    bank,
    &[a.clone(), b.clone()],
    &config(4, 3_600, 2),
    ts(1),
  )
  .await
  .expect("first");
  let second = build_temporal_links(
    &store,
    bank,
    &[a.clone(), b.clone()],
    &config(4, 3_600, 2),
    ts(1),
  )
  .await
  .expect("second");

  assert_eq!(pairs(&first), pairs(&second));
  let first_weights: Vec<f32> = first.iter().map(|link| link.weight).collect();
  let second_weights: Vec<f32> = second.iter().map(|link| link.weight).collect();
  assert_eq!(first_weights, second_weights);
}
