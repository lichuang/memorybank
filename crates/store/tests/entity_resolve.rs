//! P1-9: entity existence, candidate retrieval, and resolution ρ.

use engine::ports::{NewEntity, Store};
use engine::resolve::{MentionContext, ResolveWeights, resolve};
use engine::testing::timestamp;
use engine::types::{Bank, BankId, BankProfile, EntityId, EntityKind, EntityMention, FactId};
use store::SqliteStore;
use store::testing::memory_pool;

fn ts(millis: i64) -> engine::types::Timestamp {
  timestamp(millis)
}

async fn setup() -> (SqliteStore, BankId) {
  let store = SqliteStore::new(memory_pool().await.expect("in-memory store"));
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

fn entity(bank: BankId, name: &str, kind: EntityKind) -> NewEntity {
  NewEntity {
    id: EntityId::new(),
    bank_id: bank,
    canonical_name: name.into(),
    entity_type: kind,
    created_at: ts(1_000),
  }
}

async fn ensure(store: &SqliteStore, bank: BankId, name: &str) -> EntityId {
  let new = entity(bank, name, EntityKind::Person);
  store.ensure_entity(new).await.expect("ensure entity").0
}

async fn bind_mention(
  store: &SqliteStore,
  bank: BankId,
  entity: EntityId,
  fact: FactId,
  mention: &str,
) {
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, created_at, updated_at) \
     VALUES (?, ?, 'world', 'fact', 0, 0)",
  )
  .bind(fact.to_string())
  .bind(bank.to_string())
  .execute(store.pool())
  .await
  .expect("insert fact");

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
  .expect("insert binding");
}

#[tokio::test]
async fn ensure_entity_creates_then_reuses() {
  let (store, bank) = setup().await;
  let first = entity(bank, "emily", EntityKind::Person);
  let id = first.id;

  let (created, is_new) = store.ensure_entity(first).await.expect("create");
  assert_eq!(created, id);
  assert!(is_new);

  let (reused, is_new) = store
    .ensure_entity(entity(bank, "emily", EntityKind::Organization))
    .await
    .expect("conflict");
  assert_eq!(reused, id);
  assert!(!is_new, "the existing row wins");

  let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entity WHERE bank_id = ?")
    .bind(bank.to_string())
    .fetch_one(store.pool())
    .await
    .expect("count");
  assert_eq!(count, 1);

  let kind: String = sqlx::query_scalar("SELECT entity_type FROM entity WHERE id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("kind");
  assert_eq!(kind, "person", "the conflict does not overwrite the type");
}

#[tokio::test]
async fn ensure_entity_rejects_an_empty_name() {
  let (store, bank) = setup().await;

  let error = store
    .ensure_entity(entity(bank, "   ", EntityKind::Other))
    .await
    .expect_err("empty name");

  assert!(matches!(error, engine::Error::Config(_)));
}

#[tokio::test]
async fn candidates_match_exact_names_and_variants() {
  let (store, bank) = setup().await;
  let emily = ensure(&store, bank, "emily smith").await;
  ensure(&store, bank, "bob").await;
  bind_mention(&store, bank, emily, FactId::new(), "Emily S.").await;

  let exact = store
    .entity_candidates(bank, "emily smith", 10)
    .await
    .expect("exact");
  assert_eq!(
    exact[0].entity.id, emily,
    "exact matches are returned first"
  );

  let variant = store
    .entity_candidates(bank, "emily s.", 10)
    .await
    .expect("variant");
  let emily_candidate = variant
    .iter()
    .find(|candidate| candidate.entity.id == emily)
    .expect("variant match");
  assert!(
    emily_candidate.variants.contains(&"Emily S.".to_string()),
    "variants: {:?}",
    emily_candidate.variants
  );

  let limited = store
    .entity_candidates(bank, "nobody", 1)
    .await
    .expect("limited");
  assert_eq!(limited.len(), 1, "the limit is a hard cap");
}

#[tokio::test]
async fn candidates_include_co_mentioned_entities() {
  let (store, bank) = setup().await;
  let alice = ensure(&store, bank, "alice").await;
  let bob = ensure(&store, bank, "bob").await;
  bind_mention(&store, bank, alice, FactId::new(), "Alice").await;
  // Both entities appear in the same fact.
  let fact = FactId::new();
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, created_at, updated_at) \
     VALUES (?, ?, 'world', 'fact', 0, 0)",
  )
  .bind(fact.to_string())
  .bind(bank.to_string())
  .execute(store.pool())
  .await
  .expect("insert fact");
  for (entity, mention) in [(alice, "Alice"), (bob, "Bob")] {
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
    .expect("insert binding");
  }

  let candidates = store
    .entity_candidates(bank, "alice", 10)
    .await
    .expect("candidates");
  let alice_candidate = candidates
    .iter()
    .find(|candidate| candidate.entity.id == alice)
    .expect("alice candidate");

  assert!(
    alice_candidate.co_mentioned.contains(&bob),
    "co-mentioned: {:?}",
    alice_candidate.co_mentioned
  );
}

#[tokio::test]
async fn candidates_are_ordered_by_mention_count_and_capped() {
  let (store, bank) = setup().await;
  let first = ensure(&store, bank, "first").await;
  let second = ensure(&store, bank, "second").await;
  let third = ensure(&store, bank, "third").await;

  for (id, count) in [(first, 5), (second, 1), (third, 3)] {
    sqlx::query("UPDATE entity SET mention_count = ? WHERE id = ?")
      .bind(count)
      .bind(id.to_string())
      .execute(store.pool())
      .await
      .expect("update count");
  }

  let candidates = store
    .entity_candidates(bank, "unknown", 2)
    .await
    .expect("candidates");

  assert_eq!(candidates.len(), 2);
  assert_eq!(candidates[0].entity.id, first);
  assert_eq!(candidates[1].entity.id, third);
}

#[tokio::test]
async fn resolve_prefers_an_exact_canonical_name() {
  let (store, bank) = setup().await;
  let existing = ensure(&store, bank, "emily smith").await;

  let id = resolve(
    &store,
    bank,
    &EntityMention {
      text: "  Emily   Smith ".into(),
      kind: Some(EntityKind::Person),
    },
    &MentionContext::default(),
    &ResolveWeights::default(),
    ts(5_000),
  )
  .await
  .expect("resolve");

  assert_eq!(id, existing);
}

#[tokio::test]
async fn resolve_fuzzy_matches_a_typo() {
  let (store, bank) = setup().await;
  let existing = ensure(&store, bank, "emily smith").await;

  let id = resolve(
    &store,
    bank,
    &EntityMention {
      text: "emily smyth".into(),
      kind: None,
    },
    &MentionContext::default(),
    &ResolveWeights::default(),
    ts(5_000),
  )
  .await
  .expect("resolve");

  assert_eq!(id, existing, "10/11 similarity is above the threshold");
}

#[tokio::test]
async fn resolve_creates_an_unknown_entity() {
  let (store, bank) = setup().await;

  let id = resolve(
    &store,
    bank,
    &EntityMention {
      text: "David".into(),
      kind: Some(EntityKind::Organization),
    },
    &MentionContext::default(),
    &ResolveWeights::default(),
    ts(5_000),
  )
  .await
  .expect("resolve");

  let row: (String, String, i64) =
    sqlx::query_as("SELECT canonical_name, entity_type, created_at FROM entity WHERE id = ?")
      .bind(id.to_string())
      .fetch_one(store.pool())
      .await
      .expect("entity row");
  assert_eq!(row.0, "david");
  assert_eq!(row.1, "organization");
  assert_eq!(row.2, 5_000);
}

#[tokio::test]
async fn resolve_rejects_an_empty_mention() {
  let (store, bank) = setup().await;

  let error = resolve(
    &store,
    bank,
    &EntityMention {
      text: " \n\t ".into(),
      kind: None,
    },
    &MentionContext::default(),
    &ResolveWeights::default(),
    ts(5_000),
  )
  .await
  .expect_err("empty mention");

  assert!(matches!(error, engine::Error::Config(_)));
}

#[tokio::test]
async fn resolve_below_the_threshold_creates_a_new_entity() {
  let (store, bank) = setup().await;
  ensure(&store, bank, "emily smith").await;

  let id = resolve(
    &store,
    bank,
    &EntityMention {
      text: "emily jones".into(),
      kind: None,
    },
    &MentionContext::default(),
    &ResolveWeights::default(),
    ts(5_000),
  )
  .await
  .expect("resolve");

  let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entity WHERE bank_id = ?")
    .bind(bank.to_string())
    .fetch_one(store.pool())
    .await
    .expect("count");
  assert_eq!(count, 2);
  let created: String = sqlx::query_scalar("SELECT canonical_name FROM entity WHERE id = ?")
    .bind(id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("created entity");
  assert_eq!(created, "emily jones");
}

#[tokio::test]
async fn concurrent_resolution_of_the_same_name_yields_one_entity() {
  let (store, bank) = setup().await;
  let weights = ResolveWeights::default();
  let ctx = MentionContext::default();

  let mut handles = Vec::new();
  for _ in 0..10 {
    let store = store.clone();
    let mention = EntityMention {
      text: "david".into(),
      kind: None,
    };
    let weights = weights.clone();
    let ctx = ctx.clone();
    handles.push(tokio::spawn(async move {
      resolve(&store, bank, &mention, &ctx, &weights, ts(5_000))
        .await
        .expect("resolve")
    }));
  }

  let mut ids = Vec::new();
  for handle in handles {
    ids.push(handle.await.expect("join"));
  }

  assert!(
    ids.windows(2).all(|pair| pair[0] == pair[1]),
    "ids: {ids:?}"
  );
  let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entity WHERE bank_id = ?")
    .bind(bank.to_string())
    .fetch_one(store.pool())
    .await
    .expect("count");
  assert_eq!(count, 1);
}
