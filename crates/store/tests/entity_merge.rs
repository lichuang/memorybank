//! Entity merge: bindings, statistics, provenance, and bank isolation.

use engine::ports::{NewEntity, Store};
use engine::testing::timestamp;
use engine::types::{Bank, BankId, BankProfile, EntityId, EntityKind, FactId};
use store::SqliteStore;
use store::testing::memory_pool;

fn ts(millis: i64) -> engine::types::Timestamp {
  timestamp(millis)
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

async fn ensure(store: &SqliteStore, bank: BankId, name: &str, kind: EntityKind) -> EntityId {
  let id = EntityId::new();
  store
    .ensure_entity(NewEntity {
      id,
      bank_id: bank,
      canonical_name: name.into(),
      entity_type: kind,
      created_at: ts(0),
    })
    .await
    .expect("ensure entity");
  id
}

async fn insert_fact(
  store: &SqliteStore,
  bank: BankId,
  fact_type: &str,
  mentioned_at: Option<i64>,
) -> FactId {
  let id = FactId::new();
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, chunk_id, occurred_start, occurred_end, \
     mentioned_at, confidence, access_count, attrs, created_at, updated_at) \
     VALUES (?, ?, ?, 'fact', NULL, NULL, NULL, ?, NULL, 0, NULL, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .bind(fact_type)
  .bind(mentioned_at)
  .execute(store.pool())
  .await
  .expect("insert fact");
  id
}

async fn bind(store: &SqliteStore, bank: BankId, fact: FactId, entity: EntityId, mention: &str) {
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
  .expect("bind");
}

#[tokio::test]
async fn merges_bindings_and_recomputes_statistics() {
  let (store, bank) = setup().await;
  let source = ensure(&store, bank, "emily", EntityKind::Person).await;
  let target = ensure(&store, bank, "emily smith", EntityKind::Other).await;

  sqlx::query("UPDATE entity SET attrs = ? WHERE id = ?")
    .bind(r#"{"note":"target","merged_from":["older"]}"#)
    .bind(target.to_string())
    .execute(store.pool())
    .await
    .expect("target attrs");
  sqlx::query("UPDATE entity SET attrs = ? WHERE id = ?")
    .bind(r#"{"merged_from":["ancient"]}"#)
    .bind(source.to_string())
    .execute(store.pool())
    .await
    .expect("source attrs");

  let first = insert_fact(&store, bank, "world", Some(1_000)).await;
  let second = insert_fact(&store, bank, "world", Some(3_000)).await;
  let shared = insert_fact(&store, bank, "world", Some(2_000)).await;
  let observation = insert_fact(&store, bank, "observation", Some(9_000)).await;

  bind(&store, bank, first, source, "Emily").await;
  bind(&store, bank, second, target, "Emily Smith").await;
  // Both entities were bound to the same fact; merging must keep one row.
  bind(&store, bank, shared, source, "Emily").await;
  bind(&store, bank, shared, target, "Emily Smith").await;
  // Observation bindings move but do not count toward mention_count.
  bind(&store, bank, observation, source, "Emily").await;

  store
    .merge_entities(bank, source, target, ts(5_000))
    .await
    .expect("merge");

  let source_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entity WHERE id = ?")
    .bind(source.to_string())
    .fetch_one(store.pool())
    .await
    .expect("source count");
  assert_eq!(source_count, 0, "the source row is deleted");

  let (mention_count, first_seen, last_seen, entity_type, updated_at, attrs): (
    i64,
    Option<i64>,
    Option<i64>,
    String,
    i64,
    Option<String>,
  ) = sqlx::query_as(
    "SELECT mention_count, first_seen, last_seen, entity_type, updated_at, attrs \
     FROM entity WHERE id = ?",
  )
  .bind(target.to_string())
  .fetch_one(store.pool())
  .await
  .expect("target");

  assert_eq!(mention_count, 3, "world bindings only, deduplicated");
  assert_eq!(first_seen, Some(1_000));
  assert_eq!(last_seen, Some(3_000));
  assert_eq!(entity_type, "person", "Other yields to the specific type");
  assert_eq!(updated_at, 5_000);

  let attrs = attrs.expect("attrs");
  let attrs: serde_json::Value = serde_json::from_str(&attrs).expect("attrs json");
  assert_eq!(
    attrs["note"], "target",
    "existing object attrs are preserved"
  );
  assert_eq!(
    attrs["merged_from"],
    serde_json::json!(["older", "ancient", source.to_string()])
  );

  let target_bindings: i64 =
    sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity WHERE entity_id = ?")
      .bind(target.to_string())
      .fetch_one(store.pool())
      .await
      .expect("target bindings");
  assert_eq!(target_bindings, 4, "three facts plus the observation");

  let source_bindings: i64 =
    sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity WHERE entity_id = ?")
      .bind(source.to_string())
      .fetch_one(store.pool())
      .await
      .expect("source bindings");
  assert_eq!(source_bindings, 0);

  let shared_bindings: i64 =
    sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity WHERE fact_id = ? AND entity_id = ?")
      .bind(shared.to_string())
      .bind(target.to_string())
      .fetch_one(store.pool())
      .await
      .expect("shared binding");
  assert_eq!(shared_bindings, 1, "composite primary key stays unique");
}

#[tokio::test]
async fn rejects_self_merge() {
  let (store, bank) = setup().await;
  let id = ensure(&store, bank, "emily", EntityKind::Person).await;

  let error = store
    .merge_entities(bank, id, id, ts(1_000))
    .await
    .expect_err("self merge");

  assert!(matches!(error, engine::Error::Config(_)), "error: {error}");
}

#[tokio::test]
async fn rejects_missing_or_cross_bank_entities() {
  let (store, bank) = setup().await;
  let target = ensure(&store, bank, "emily smith", EntityKind::Person).await;
  let missing = EntityId::new();

  let error = store
    .merge_entities(bank, missing, target, ts(1_000))
    .await
    .expect_err("missing source");
  assert!(matches!(error, engine::Error::NotFound("entity", _)));

  let other_bank = Bank {
    id: BankId::new(),
    name: "other".into(),
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
  store.create_bank(other_bank.clone()).await.expect("bank");
  let foreign = ensure(&store, other_bank.id, "emily", EntityKind::Person).await;

  let error = store
    .merge_entities(bank, foreign, target, ts(1_000))
    .await
    .expect_err("cross-bank source");
  assert!(matches!(error, engine::Error::NotFound("entity", _)));
}
