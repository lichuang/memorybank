//! Graph adjacency and spreading activation over a real store.

use engine::ports::{FactScope, NeighborKind, Store};
use engine::recall::{GraphConfig, spread};
use engine::types::{Bank, BankId, BankProfile, EntityId, FactId, FactType, LinkId};
use sqlx::SqlitePool;
use store::SqliteStore;
use store::testing::memory_pool;

fn ts(millis: i64) -> engine::types::Timestamp {
  engine::testing::timestamp(millis)
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

async fn insert_fact(pool: &SqlitePool, bank: BankId, fact_type: FactType, text: &str) -> FactId {
  let id = FactId::new();
  let confidence = (fact_type == FactType::Opinion).then_some(0.5_f64);
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, confidence, created_at, updated_at) \
     VALUES (?, ?, ?, ?, ?, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(fact_type.as_str())
  .bind(text)
  .bind(confidence)
  .execute(pool)
  .await
  .expect("fact");
  id
}

async fn insert_entity(pool: &SqlitePool, bank: BankId, name: &str) -> EntityId {
  let id = EntityId::new();
  sqlx::query(
    "INSERT INTO entity (id, bank_id, canonical_name, entity_type, mention_count, \
     created_at, updated_at) VALUES (?, ?, ?, 'person', 0, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(name)
  .execute(pool)
  .await
  .expect("entity");
  id
}

async fn bind(pool: &SqlitePool, bank: BankId, fact: FactId, entity: EntityId) {
  sqlx::query(
    "INSERT INTO fact_entity (bank_id, fact_id, entity_id, mention_text, created_at, updated_at) \
     VALUES (?, ?, ?, NULL, 0, 0)",
  )
  .bind(bank.to_string())
  .bind(fact.to_string())
  .bind(entity.to_string())
  .execute(pool)
  .await
  .expect("binding");
}

async fn insert_link(
  pool: &SqlitePool,
  bank: BankId,
  from: FactId,
  to: FactId,
  link_type: &str,
  relation: Option<&str>,
  weight: f64,
) {
  sqlx::query(
    "INSERT INTO fact_link (id, bank_id, from_fact_id, to_fact_id, link_type, relation, \
     weight, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)",
  )
  .bind(LinkId::new().to_string())
  .bind(bank.to_string())
  .bind(from.to_string())
  .bind(to.to_string())
  .bind(link_type)
  .bind(relation)
  .bind(weight)
  .execute(pool)
  .await
  .expect("link");
}

fn config() -> GraphConfig {
  GraphConfig {
    delta: 0.5,
    mu_temporal: 0.5,
    mu_semantic: 1.0,
    mu_entity: 1.5,
    mu_causal: 1.5,
    max_depth: 10,
    max_visits: 100,
  }
}

#[tokio::test]
async fn neighbors_merge_links_and_derived_entity_adjacency() {
  let (store, bank) = setup().await;
  let from = insert_fact(store.pool(), bank, FactType::World, "from").await;
  let opinion = insert_fact(store.pool(), bank, FactType::Opinion, "opinion").await;
  let temporal = insert_fact(store.pool(), bank, FactType::World, "temporal").await;
  let shared = insert_fact(store.pool(), bank, FactType::World, "shared").await;
  let entity = insert_entity(store.pool(), bank, "alice").await;

  insert_link(store.pool(), bank, from, opinion, "semantic", None, 0.8).await;
  insert_link(
    store.pool(),
    bank,
    from,
    opinion,
    "causal",
    Some("causes"),
    1.0,
  )
  .await;
  insert_link(
    store.pool(),
    bank,
    from,
    opinion,
    "causal",
    Some("enables"),
    1.0,
  )
  .await;
  insert_link(store.pool(), bank, from, temporal, "temporal", None, 0.4).await;
  bind(store.pool(), bank, from, entity).await;
  bind(store.pool(), bank, shared, entity).await;

  let neighbors = store
    .neighbors(bank, from, &FactScope::All)
    .await
    .expect("neighbors");

  assert_eq!(
    neighbors.len(),
    4,
    "one semantic, one causal, one temporal, one entity"
  );
  assert!(neighbors.iter().any(|neighbor| {
    neighbor.fact_id == opinion
      && neighbor.kind == NeighborKind::Semantic
      && (neighbor.weight - 0.8).abs() < 1e-6
  }));
  assert_eq!(
    neighbors
      .iter()
      .filter(|neighbor| neighbor.fact_id == opinion && neighbor.kind == NeighborKind::Causal)
      .count(),
    1,
    "causal relations of one kind collapse to the strongest edge"
  );
  assert!(neighbors.iter().any(|neighbor| {
    neighbor.fact_id == temporal
      && neighbor.kind == NeighborKind::Temporal
      && (neighbor.weight - 0.4).abs() < 1e-6
  }));
  assert!(neighbors.iter().any(|neighbor| {
    neighbor.fact_id == shared && neighbor.kind == NeighborKind::Entity && neighbor.weight == 1.0
  }));

  let again = store
    .neighbors(bank, from, &FactScope::All)
    .await
    .expect("neighbors again");
  assert_eq!(neighbors, again, "ordering is deterministic");
}

#[tokio::test]
async fn neighbors_respect_scope_and_empty_types() {
  let (store, bank) = setup().await;
  let from = insert_fact(store.pool(), bank, FactType::World, "from").await;
  let opinion = insert_fact(store.pool(), bank, FactType::Opinion, "opinion").await;
  let world = insert_fact(store.pool(), bank, FactType::World, "world").await;
  insert_link(store.pool(), bank, from, opinion, "semantic", None, 0.8).await;
  insert_link(store.pool(), bank, from, world, "semantic", None, 0.8).await;

  let worlds = store
    .neighbors(bank, from, &FactScope::Types(vec![FactType::World]))
    .await
    .expect("worlds");
  assert_eq!(
    worlds
      .iter()
      .map(|neighbor| neighbor.fact_id)
      .collect::<Vec<_>>(),
    vec![world]
  );

  let opinions = store
    .neighbors(bank, from, &FactScope::Types(vec![FactType::Opinion]))
    .await
    .expect("opinions");
  assert_eq!(
    opinions
      .iter()
      .map(|neighbor| neighbor.fact_id)
      .collect::<Vec<_>>(),
    vec![opinion]
  );

  let none = store
    .neighbors(bank, from, &FactScope::Types(Vec::new()))
    .await
    .expect("none");
  assert!(none.is_empty());
}

#[tokio::test]
async fn spread_decays_along_a_chain_and_terminates_cycles() {
  let (store, bank) = setup().await;
  let a = insert_fact(store.pool(), bank, FactType::World, "a").await;
  let b = insert_fact(store.pool(), bank, FactType::World, "b").await;
  let c = insert_fact(store.pool(), bank, FactType::World, "c").await;
  insert_link(store.pool(), bank, a, b, "semantic", None, 0.8).await;
  insert_link(store.pool(), bank, b, c, "semantic", None, 0.5).await;
  // A cycle must not keep increasing A (0.4 * 0.8 * 0.5 = 0.16).
  insert_link(store.pool(), bank, b, a, "semantic", None, 0.8).await;

  let results = spread(&store, bank, &[(a, 1.0)], &config())
    .await
    .expect("spread");
  let scores: std::collections::HashMap<FactId, f32> = results.into_iter().collect();

  assert!((scores[&a] - 1.0).abs() < 1e-6);
  assert!((scores[&b] - 0.4).abs() < 1e-6);
  assert!((scores[&c] - 0.1).abs() < 1e-6);

  let again = spread(&store, bank, &[(a, 1.0)], &config())
    .await
    .expect("spread again");
  assert_eq!(
    scores,
    again.into_iter().collect(),
    "activation is deterministic"
  );
}

#[tokio::test]
async fn spread_honors_depth_and_visit_caps() {
  let (store, bank) = setup().await;
  let a = insert_fact(store.pool(), bank, FactType::World, "a").await;
  let b = insert_fact(store.pool(), bank, FactType::World, "b").await;
  let c = insert_fact(store.pool(), bank, FactType::World, "c").await;
  insert_link(store.pool(), bank, a, b, "semantic", None, 0.8).await;
  insert_link(store.pool(), bank, b, c, "semantic", None, 0.8).await;

  let shallow = GraphConfig {
    max_depth: 1,
    ..config()
  };
  let depth_limited = spread(&store, bank, &[(a, 1.0)], &shallow)
    .await
    .expect("depth");
  let ids: Vec<FactId> = depth_limited.iter().map(|(id, _)| *id).collect();
  assert!(ids.contains(&a));
  assert!(ids.contains(&b));
  assert!(!ids.contains(&c));

  let one_visit = GraphConfig {
    max_visits: 1,
    ..config()
  };
  let visit_limited = spread(&store, bank, &[(a, 1.0)], &one_visit)
    .await
    .expect("visits");
  let ids: Vec<FactId> = visit_limited.iter().map(|(id, _)| *id).collect();
  assert!(ids.contains(&a));
  assert!(ids.contains(&b));
  assert!(!ids.contains(&c), "B is discovered but not expanded");
}

#[tokio::test]
async fn spread_uses_derived_entity_adjacency() {
  let (store, bank) = setup().await;
  let a = insert_fact(store.pool(), bank, FactType::World, "a").await;
  let c = insert_fact(store.pool(), bank, FactType::World, "c").await;
  let entity = insert_entity(store.pool(), bank, "alice").await;
  bind(store.pool(), bank, a, entity).await;
  bind(store.pool(), bank, c, entity).await;

  let results = spread(&store, bank, &[(a, 1.0)], &config())
    .await
    .expect("entity spread");
  let scores: std::collections::HashMap<FactId, f32> = results.into_iter().collect();

  assert!(
    (scores[&c] - 0.75).abs() < 1e-6,
    "delta 0.5 * mu_entity 1.5"
  );
  assert_eq!(scores.len(), 2, "the seed plus the entity neighbour");
}

#[tokio::test]
async fn spread_ignores_empty_and_non_positive_seeds() {
  let (store, bank) = setup().await;
  let a = insert_fact(store.pool(), bank, FactType::World, "a").await;

  assert!(
    spread(&store, bank, &[], &config())
      .await
      .expect("empty")
      .is_empty()
  );
  assert!(
    spread(&store, bank, &[(a, 0.0)], &config())
      .await
      .expect("zero")
      .is_empty()
  );
  assert!(
    spread(&store, bank, &[(a, f32::NAN)], &config())
      .await
      .expect("nan")
      .is_empty()
  );
}
