//! Opinion entity binding and persistence against a real store.

use engine::ports::{NewEntity, Store};
use engine::reflect::{bind_known_entities, prepare_opinion};
use engine::testing::timestamp;
use engine::types::{Bank, BankId, BankProfile, EntityId, EntityKind, FactId};
use store::testing::memory_pool;
use store::{EMBEDDING_DIM, SqliteStore};

fn ts(millis: i64) -> engine::types::Timestamp {
  timestamp(millis)
}

fn embedding() -> Vec<f32> {
  let mut values = vec![0.0; EMBEDDING_DIM];
  values[0] = 1.0;
  values
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
  store.create_bank(bank.clone()).await.expect("bank");
  (store, bank.id)
}

async fn entity(store: &SqliteStore, bank: BankId, name: &str) -> EntityId {
  let id = EntityId::new();
  store
    .ensure_entity(NewEntity {
      id,
      bank_id: bank,
      canonical_name: name.into(),
      entity_type: EntityKind::Person,
      created_at: ts(0),
    })
    .await
    .expect("entity");
  id
}

/// Adds a binding so the entity also owns a mention variant.
async fn variant(store: &SqliteStore, bank: BankId, entity: EntityId, mention: &str) {
  let fact = FactId::new();
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, created_at, updated_at) \
     VALUES (?, ?, 'world', 'fact', 0, 0)",
  )
  .bind(fact.to_string())
  .bind(bank.to_string())
  .execute(store.pool())
  .await
  .expect("fact");
  sqlx::query(
    "INSERT INTO fact_entity (bank_id, fact_id, entity_id, mention_text, created_at, updated_at) \
     VALUES (?, ?, ?, ?, 0, 0)",
  )
  .bind(bank.to_string())
  .bind(fact.to_string())
  .bind(entity.to_string())
  .bind(mention)
  .execute(store.pool())
  .await
  .expect("binding");
}

#[tokio::test]
async fn binds_two_entities_named_in_the_opinion() {
  let (store, bank) = setup().await;
  let alice = entity(&store, bank, "alice").await;
  let bob = entity(&store, bank, "bob").await;
  entity(&store, bank, "carol").await;

  let bindings = bind_known_entities(
    &store,
    bank,
    "I think Alice and Bob collaborate better than most teams",
  )
  .await
  .expect("bind");

  assert_eq!(bindings.len(), 2);
  let ids: Vec<EntityId> = bindings.iter().map(|(id, _)| *id).collect();
  assert!(ids.contains(&alice));
  assert!(ids.contains(&bob));
}

#[tokio::test]
async fn binds_nothing_when_the_opinion_names_no_known_entity() {
  let (store, bank) = setup().await;
  entity(&store, bank, "alice").await;

  let bindings = bind_known_entities(&store, bank, "I prefer statically typed languages")
    .await
    .expect("bind");

  assert!(bindings.is_empty());
}

#[tokio::test]
async fn matches_recorded_variants_case_insensitively() {
  let (store, bank) = setup().await;
  let alice = entity(&store, bank, "alice").await;
  variant(&store, bank, alice, "Alice S.").await;

  let bindings = bind_known_entities(&store, bank, "I trust ALICE S. with production access")
    .await
    .expect("bind");

  assert_eq!(bindings.len(), 1);
  assert_eq!(bindings[0].0, alice);
  assert_eq!(bindings[0].1.as_deref(), Some("alice s."));
}

#[tokio::test]
async fn persists_the_opinion_with_its_bindings_and_embedding() {
  let (store, bank) = setup().await;
  let alice = entity(&store, bank, "alice").await;

  let id = FactId::new();
  let opinion = prepare_opinion(
    &store,
    id,
    bank,
    "I believe Alice is reliable",
    0.7,
    Some(embedding()),
    ts(9_000),
  )
  .await
  .expect("prepare");
  store.insert_opinion(opinion).await.expect("insert");

  let (fact_type, chunk_id, confidence, text): (String, Option<String>, f64, String) =
    sqlx::query_as("SELECT fact_type, chunk_id, confidence, text FROM fact WHERE id = ?")
      .bind(id.to_string())
      .fetch_one(store.pool())
      .await
      .expect("fact");
  assert_eq!(fact_type, "opinion");
  assert_eq!(chunk_id, None, "reflect opinions have no chunk");
  assert!((confidence - 0.7).abs() < 1e-6);
  assert_eq!(text, "I believe Alice is reliable");

  let vectors: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_vec WHERE fact_id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("vector");
  assert_eq!(vectors, 1);

  let fts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_fts WHERE fact_id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("fts");
  assert_eq!(fts, 1);

  let bound: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity WHERE fact_id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("bindings");
  assert_eq!(bound, 1);

  let bound_entity: String =
    sqlx::query_scalar("SELECT entity_id FROM fact_entity WHERE fact_id = ?")
      .bind(id.to_string())
      .fetch_one(store.pool())
      .await
      .expect("entity id");
  assert_eq!(bound_entity, alice.to_string());
}

#[tokio::test]
async fn duplicate_bindings_collapse_to_one_row() {
  let (store, bank) = setup().await;
  let alice = entity(&store, bank, "alice").await;

  let opinion = prepare_opinion(
    &store,
    FactId::new(),
    bank,
    "I trust Alice because Alice delivers",
    0.6,
    None,
    ts(1),
  )
  .await
  .expect("prepare");
  let id = opinion.id;
  store.insert_opinion(opinion).await.expect("insert");

  let bound: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity WHERE fact_id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("bindings");
  assert_eq!(bound, 1, "one entity, one binding row");

  let stored: String = sqlx::query_scalar("SELECT entity_id FROM fact_entity WHERE fact_id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("entity id");
  assert_eq!(stored, alice.to_string());
}

#[tokio::test]
async fn rejects_an_out_of_range_confidence() {
  let (store, bank) = setup().await;

  let opinion = engine::ports::NewOpinion {
    id: FactId::new(),
    bank_id: bank,
    text: "I think A".into(),
    confidence: 1.5,
    entities: Vec::new(),
    embedding: None,
    created_at: ts(0),
  };

  let error = store
    .insert_opinion(opinion)
    .await
    .expect_err("out of range");
  assert!(matches!(error, engine::Error::Config(_)), "error: {error}");

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact")
    .fetch_one(store.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 0, "nothing is written when validation fails");
}
