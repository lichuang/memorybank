//! P1-16: `commit_chunk` (tx2) — atomicity and claim-token guarding.

use std::time::Duration;

use engine::ports::{
  ChunkCommit, ClaimedChunk, EntityStatUpdate, NewBinding, NewChunk, NewEntity, NewFact, NewLink,
  OpinionConfidenceUpdate, Store,
};
use engine::testing::timestamp;
use engine::types::{
  Bank, BankId, BankProfile, CausalRelation, ChunkId, Entity, EntityId, EntityKind, FactId, LinkId,
  LinkType, OccurrenceInterval, Source, SourceId, SourceStatus,
};
use sqlx::SqlitePool;
use store::testing::memory_pool;
use store::{EMBEDDING_DIM, SqliteStore};

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

async fn claimed_chunk(store: &SqliteStore, bank: BankId) -> ClaimedChunk {
  let source = Source {
    id: SourceId::new(),
    bank_id: bank,
    title: None,
    payload_text: "payload".into(),
    status: SourceStatus::Received,
    created_at: ts(0),
    updated_at: ts(0),
  };
  store.create_source(source.clone()).await.expect("source");

  store
    .append_chunks(
      source.id,
      vec![NewChunk {
        id: ChunkId::new(),
        bank_id: bank,
        source_id: source.id,
        chunk_index: 0,
        text: "chunk".into(),
        content_hash: "hash".into(),
        time_start: None,
        time_end: None,
        created_at: ts(0),
      }],
    )
    .await
    .expect("chunks");

  store
    .claim_chunk(bank, ts(1_000), Duration::from_secs(3_600))
    .await
    .expect("claim")
    .expect("pending chunk")
}

fn vector() -> Vec<f32> {
  let mut values = vec![0.0; EMBEDDING_DIM];
  values[0] = 1.0;
  values
}

fn entity(bank: BankId, name: &str) -> Entity {
  Entity {
    id: EntityId::new(),
    bank_id: bank,
    canonical_name: name.into(),
    entity_type: EntityKind::Person,
    mention_count: 0,
    first_seen: None,
    last_seen: None,
    attrs: None,
    created_at: ts(0),
    updated_at: ts(0),
  }
}

fn fact(bank: BankId, chunk: ChunkId, text: &str, embedding: Option<Vec<f32>>) -> NewFact {
  NewFact {
    id: FactId::new(),
    bank_id: bank,
    fact_type: engine::types::FactType::World,
    text: text.into(),
    chunk_id: Some(chunk),
    embedding,
    occurred: OccurrenceInterval::Unknown,
    mentioned_at: None,
    confidence: None,
    attrs: None,
    created_at: ts(1_000),
  }
}

async fn opinion(pool: &SqlitePool, bank: BankId) -> FactId {
  let id = FactId::new();
  sqlx::query(
    "INSERT INTO fact (id, bank_id, fact_type, text, chunk_id, confidence, created_at, updated_at) \
     VALUES (?, ?, 'opinion', 'old opinion', NULL, 0.5, 0, 0)",
  )
  .bind(id.to_string())
  .bind(bank.to_string())
  .execute(pool)
  .await
  .expect("insert opinion");
  id
}

async fn count(pool: &SqlitePool, sql: &str) -> i64 {
  sqlx::query_scalar(sql)
    .fetch_one(pool)
    .await
    .expect("count")
}

fn full_commit(
  bank: BankId,
  claimed: &ClaimedChunk,
  opinion_id: FactId,
) -> (ChunkCommit, FactId, FactId, EntityId) {
  let emily = entity(bank, "emily");
  let first = fact(
    bank,
    claimed.chunk.id,
    "emily joined google",
    Some(vector()),
  );
  let second = fact(bank, claimed.chunk.id, "google is a company", None);

  let commit = ChunkCommit {
    chunk_id: claimed.chunk.id,
    claimed_at: claimed.chunk.updated_at,
    completed_at: ts(2_000),
    facts: vec![first.clone(), second.clone()],
    entities: vec![emily.clone()],
    bindings: vec![NewBinding {
      bank_id: bank,
      fact_id: first.id,
      entity_id: emily.id,
      mention_text: Some("Emily".into()),
      created_at: ts(1_000),
    }],
    links: vec![
      NewLink {
        id: LinkId::new(),
        bank_id: bank,
        from: first.id,
        to: second.id,
        link_type: LinkType::Semantic,
        relation: None,
        weight: 0.9,
        created_at: ts(1_000),
      },
      NewLink {
        id: LinkId::new(),
        bank_id: bank,
        from: first.id,
        to: second.id,
        link_type: LinkType::Causal,
        relation: Some(CausalRelation::Causes),
        weight: 1.0,
        created_at: ts(1_000),
      },
    ],
    entity_stats: vec![EntityStatUpdate {
      entity_id: emily.id,
      add: true,
      at: Some(ts(1_000)),
      updated_at: ts(1_000),
    }],
    opinion_updates: vec![OpinionConfidenceUpdate {
      opinion_id,
      new_confidence: 0.8,
      new_text: Some("revised opinion".into()),
      updated_at: ts(2_000),
    }],
  };

  (commit, first.id, second.id, emily.id)
}

#[tokio::test]
async fn commits_every_product_and_flips_the_chunk() {
  let (store, bank) = setup().await;
  let claimed = claimed_chunk(&store, bank).await;
  let opinion_id = opinion(store.pool(), bank).await;
  let (commit, first, second, emily) = full_commit(bank, &claimed, opinion_id);

  store.commit_chunk(commit).await.expect("commit");

  let (status, updated_at): (String, i64) =
    sqlx::query_as("SELECT status, updated_at FROM chunk WHERE id = ?")
      .bind(claimed.chunk.id.to_string())
      .fetch_one(store.pool())
      .await
      .expect("chunk row");
  assert_eq!(status, "done");
  assert_eq!(updated_at, 2_000);

  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact WHERE chunk_id = ?")
    .bind(claimed.chunk.id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 2);
  let vectors: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_vec WHERE fact_id = ?")
    .bind(first.to_string())
    .fetch_one(store.pool())
    .await
    .expect("vector");
  assert_eq!(vectors, 1, "only the embedded fact has a vector row");

  let bindings: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity WHERE fact_id = ?")
    .bind(first.to_string())
    .fetch_one(store.pool())
    .await
    .expect("bindings");
  assert_eq!(bindings, 1);

  let links: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_link WHERE from_fact_id = ?")
    .bind(first.to_string())
    .fetch_one(store.pool())
    .await
    .expect("links");
  assert_eq!(links, 2);

  let fts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_fts WHERE fact_id IN (?, ?)")
    .bind(first.to_string())
    .bind(second.to_string())
    .fetch_one(store.pool())
    .await
    .expect("fts");
  assert_eq!(fts, 2);

  let (mention_count, first_seen, last_seen): (i64, Option<i64>, Option<i64>) =
    sqlx::query_as("SELECT mention_count, first_seen, last_seen FROM entity WHERE id = ?")
      .bind(emily.to_string())
      .fetch_one(store.pool())
      .await
      .expect("entity stats");
  assert_eq!(
    (mention_count, first_seen, last_seen),
    (1, Some(1_000), Some(1_000))
  );

  let (confidence, text, updated): (f64, String, i64) =
    sqlx::query_as("SELECT confidence, text, updated_at FROM fact WHERE id = ?")
      .bind(opinion_id.to_string())
      .fetch_one(store.pool())
      .await
      .expect("opinion");
  assert!((confidence - 0.8).abs() < 1e-6);
  assert_eq!(text, "revised opinion");
  assert_eq!(updated, 2_000);
}

#[tokio::test]
async fn rejects_a_stale_claim_and_writes_nothing() {
  let (store, bank) = setup().await;
  let claimed = claimed_chunk(&store, bank).await;
  let opinion_id = opinion(store.pool(), bank).await;
  let (mut commit, ..) = full_commit(bank, &claimed, opinion_id);
  commit.claimed_at = ts(claimed.chunk.updated_at.timestamp_millis() + 1);

  let error = store.commit_chunk(commit).await.expect_err("stale claim");

  assert!(
    matches!(error, engine::Error::Conflict(_, _)),
    "error: {error}"
  );
  assert_eq!(
    count(store.pool(), "SELECT COUNT(*) FROM fact").await,
    1,
    "only the seed opinion"
  );
  assert_eq!(count(store.pool(), "SELECT COUNT(*) FROM entity").await, 0);
  assert_eq!(
    count(store.pool(), "SELECT COUNT(*) FROM fact_link").await,
    0
  );
  let status: String = sqlx::query_scalar("SELECT status FROM chunk WHERE id = ?")
    .bind(claimed.chunk.id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("chunk");
  assert_eq!(status, "extracting");
}

#[tokio::test]
async fn replaying_a_commit_is_rejected() {
  let (store, bank) = setup().await;
  let claimed = claimed_chunk(&store, bank).await;
  let opinion_id = opinion(store.pool(), bank).await;
  let (commit, ..) = full_commit(bank, &claimed, opinion_id);

  store.commit_chunk(commit.clone()).await.expect("first");
  let error = store.commit_chunk(commit).await.expect_err("replay");

  assert!(
    matches!(error, engine::Error::Conflict(_, _)),
    "error: {error}"
  );
  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact WHERE chunk_id = ?")
    .bind(claimed.chunk.id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 2, "the rejected replay wrote nothing");
}

#[tokio::test]
async fn a_failure_rolls_back_everything() {
  let (store, bank) = setup().await;
  let claimed = claimed_chunk(&store, bank).await;
  let opinion_id = opinion(store.pool(), bank).await;
  let (mut commit, first, ..) = full_commit(bank, &claimed, opinion_id);
  // A duplicate fact id makes the second insert fail after the entity/first fact.
  let duplicate = commit.facts[0].clone();
  commit.facts.push(duplicate);

  let error = store
    .commit_chunk(commit)
    .await
    .expect_err("duplicate fact id");
  assert!(!error.to_string().is_empty());

  assert_eq!(count(store.pool(), "SELECT COUNT(*) FROM entity").await, 0);
  let facts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact WHERE id = ?")
    .bind(first.to_string())
    .fetch_one(store.pool())
    .await
    .expect("facts");
  assert_eq!(facts, 0);
  assert_eq!(
    count(store.pool(), "SELECT COUNT(*) FROM fact_vec").await,
    0
  );
  assert_eq!(
    count(store.pool(), "SELECT COUNT(*) FROM fact_entity").await,
    0
  );
  assert_eq!(
    count(store.pool(), "SELECT COUNT(*) FROM fact_link").await,
    0
  );
  assert_eq!(
    count(store.pool(), "SELECT COUNT(*) FROM fact_fts").await,
    0
  );

  let (confidence, text): (f64, String) =
    sqlx::query_as("SELECT confidence, text FROM fact WHERE id = ?")
      .bind(opinion_id.to_string())
      .fetch_one(store.pool())
      .await
      .expect("opinion");
  assert!((confidence - 0.5).abs() < 1e-6);
  assert_eq!(text, "old opinion");

  let status: String = sqlx::query_scalar("SELECT status FROM chunk WHERE id = ?")
    .bind(claimed.chunk.id.to_string())
    .fetch_one(store.pool())
    .await
    .expect("chunk");
  assert_eq!(status, "extracting");
}

#[tokio::test]
async fn rewrites_bindings_and_statistics_on_an_entity_conflict() {
  let (store, bank) = setup().await;
  let existing = store
    .ensure_entity(NewEntity {
      id: EntityId::new(),
      bank_id: bank,
      canonical_name: "emily".into(),
      entity_type: EntityKind::Person,
      created_at: ts(0),
    })
    .await
    .expect("existing entity")
    .0;

  let claimed = claimed_chunk(&store, bank).await;
  let opinion_id = opinion(store.pool(), bank).await;
  let (mut commit, first, ..) = full_commit(bank, &claimed, opinion_id);

  // The commit proposes a different id for the same canonical name.
  let candidate = EntityId::new();
  commit.entities = vec![Entity {
    id: candidate,
    bank_id: bank,
    canonical_name: "emily".into(),
    entity_type: EntityKind::Person,
    mention_count: 0,
    first_seen: None,
    last_seen: None,
    attrs: None,
    created_at: ts(0),
    updated_at: ts(0),
  }];
  commit.bindings[0].entity_id = candidate;
  commit.entity_stats[0].entity_id = candidate;

  store.commit_chunk(commit).await.expect("commit");

  let binding: String = sqlx::query_scalar("SELECT entity_id FROM fact_entity WHERE fact_id = ?")
    .bind(first.to_string())
    .fetch_one(store.pool())
    .await
    .expect("binding");
  assert_eq!(
    binding,
    existing.to_string(),
    "binding follows the existing id"
  );

  let (count, mention_count): (i64, i64) = sqlx::query_as(
    "SELECT COUNT(*), MAX(mention_count) FROM entity WHERE canonical_name = 'emily'",
  )
  .fetch_one(store.pool())
  .await
  .expect("entity");
  assert_eq!(count, 1);
  assert_eq!(mention_count, 1, "statistics follow the existing id");
}
