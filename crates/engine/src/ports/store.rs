//! Storage port: the complete persistence contract the engine depends on.
//!
//! Two rules shape every signature:
//!
//! - **IDs and time are engine-supplied.** The store never generates business
//!   IDs and never reads the wall clock; creation DTOs carry `id` and
//!   `created_at`, and methods that need "now" take it as a parameter.
//! - **No backend types leak.** Only domain types, `std::time::Duration`, and
//!   primitives cross this boundary, so a second backend can implement the same
//!   contract without changing the engine.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::types::{
  Bank, BankId, CausalRelation, Chunk, ChunkId, Entity, EntityId, EntityKind, Fact, FactId,
  FactType, LinkId, LinkType, Source, SourceId, Timestamp,
};

/// The persistence contract used by every pipeline.
#[async_trait]
pub trait Store: Send + Sync {
  // —— Bank & inbox ——

  /// Inserts a bank. Returns [`crate::Error::Conflict`] when `bank.name` exists.
  async fn create_bank(&self, bank: Bank) -> Result<()>;

  /// Loads a bank by id. Returns [`crate::Error::NotFound`] when absent.
  async fn load_bank(&self, id: BankId) -> Result<Bank>;

  /// Persists a source row (tx0 of the inbox model).
  async fn create_source(&self, source: Source) -> Result<()>;

  /// Persists chunk rows for a source (tx1), skipping any whose
  /// `(bank_id, content_hash)` already exists, and advances the source to
  /// `chunked` in the same transaction. Returns the inserted count.
  ///
  /// Every chunk must belong to `source` and to the source's bank; a mismatch is
  /// an [`crate::Error::Config`]. The source's `updated_at` becomes the latest
  /// `created_at` in the batch; an empty batch leaves it unchanged (there is
  /// still a status flip).
  async fn append_chunks(&self, source: SourceId, chunks: Vec<NewChunk>) -> Result<usize>;

  /// Atomically claims one chunk for extraction and reclaims stale
  /// `extracting` rows (`now - updated_at > stale_after`). The claimed row is
  /// returned with `status = Extracting` and `updated_at = now`.
  async fn claim_chunk(
    &self,
    bank: BankId,
    now: Timestamp,
    stale_after: Duration,
  ) -> Result<Option<ClaimedChunk>>;

  /// Returns a claimed chunk to `pending` after a failed extraction.
  async fn release_chunk(&self, chunk: ChunkId) -> Result<()>;

  /// The only transaction entry for extraction products (tx2): facts, entity
  /// bindings, links, entity statistics, and opinion updates are written
  /// together with the chunk status flip to `done`.
  async fn commit_chunk(&self, commit: ChunkCommit) -> Result<()>;

  // —— Retrieval ——

  /// Top-`k` facts by cosine similarity against `query`; rows without an
  /// embedding are skipped. Adapters may use ANN internally; the contract is
  /// "best `k` matches".
  async fn search_vector(
    &self,
    bank: BankId,
    query: &[f32],
    k: usize,
  ) -> Result<Vec<(FactId, f32)>>;

  /// Top-`k` facts by lexical (BM25) similarity to `query`.
  async fn search_bm25(&self, bank: BankId, query: &str, k: usize) -> Result<Vec<(FactId, f32)>>;

  /// Outgoing neighbors of `from`: `fact_link` edges merged with entity
  /// adjacency derived from `fact_entity` bindings (weight 1.0). The pairwise
  /// closure is never materialized.
  async fn neighbors(&self, bank: BankId, from: FactId) -> Result<Vec<Neighbor>>;

  /// Facts whose occurrence interval intersects `range` (closed interval).
  async fn facts_in_range(
    &self,
    bank: BankId,
    range: crate::types::TimeRange,
  ) -> Result<Vec<FactId>>;

  /// Loads facts by id in one round trip; order is unspecified.
  async fn fetch_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>>;

  /// Batch-increments `access_count`; does not touch `updated_at`.
  async fn increment_access(&self, ids: &[FactId]) -> Result<()>;

  // —— Entity / opinion / observation / maintenance ——

  /// Ensures an entity exists under `UNIQUE(bank_id, canonical_name)`.
  ///
  /// `new.id` is the engine-allocated candidate id. On a uniqueness conflict the
  /// existing row is reused and its id returned; the boolean reports whether a
  /// row was created.
  async fn ensure_entity(&self, new: NewEntity) -> Result<(EntityId, bool)>;

  /// Candidate entities for a mention: exact-name hits plus variants aggregated
  /// from `fact_entity.mention_text`, with the most recent mention time.
  async fn entity_candidates(&self, bank: BankId, mention: &str) -> Result<Vec<EntityCandidate>>;

  /// Writes an opinion row and its `fact_entity` bindings in one transaction.
  async fn insert_opinion(&self, opinion: NewOpinion) -> Result<()>;

  /// Appends an observation batch (rows plus bindings). Existing batches are
  /// never replaced; old batches are reclaimed by [`Store::gc_observations`].
  async fn insert_observations(&self, observations: Vec<NewObservation>) -> Result<()>;

  /// Observation rows bound to `entity`, newest first.
  async fn observations_of(
    &self,
    bank: BankId,
    entity: EntityId,
    limit: usize,
  ) -> Result<Vec<Fact>>;

  /// Derived work lists (no task-queue table): sources awaiting chunking,
  /// pending chunks, stale claims, facts missing embeddings, and entities whose
  /// evidence changed since their latest observation batch.
  async fn pending_work(
    &self,
    bank: BankId,
    now: Timestamp,
    stale_after: Duration,
  ) -> Result<WorkLists>;

  /// Writes backfilled embeddings. The engine owns the [`crate::ports::Embedder`],
  /// so the store only persists vectors; `updated_at` is left untouched.
  /// Returns the number of rows written.
  async fn update_embeddings(&self, embeddings: &[(FactId, Vec<f32>)]) -> Result<usize>;

  /// Deletes observation rows older than `now - max_age`, then keeps the newest
  /// `per_entity_cap` rows per entity binding. Returns the deleted count.
  async fn gc_observations(
    &self,
    bank: BankId,
    now: Timestamp,
    max_age: Duration,
    per_entity_cap: usize,
  ) -> Result<usize>;
}

/// A new canonical entity; the id is engine-allocated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewEntity {
  /// Candidate identifier (reused on a uniqueness conflict).
  pub id: EntityId,
  /// Owning bank.
  pub bank_id: BankId,
  /// Canonical (lowercased/trimmed) name; unique per bank.
  pub canonical_name: String,
  /// Entity category.
  pub entity_type: EntityKind,
  /// Creation time.
  pub created_at: Timestamp,
}

/// A chunk row about to be written; `status` starts at `pending`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewChunk {
  /// Engine-allocated chunk id.
  pub id: ChunkId,
  /// Owning bank.
  pub bank_id: BankId,
  /// Owning source.
  pub source_id: SourceId,
  /// Position within the source.
  pub chunk_index: u32,
  /// Verbatim slice text.
  pub text: String,
  /// Normalized hash of `text`; unique per bank.
  pub content_hash: String,
  /// Conversation-clock interval covered by the chunk.
  pub time_start: Option<Timestamp>,
  /// Conversation-clock interval covered by the chunk.
  pub time_end: Option<Timestamp>,
  /// Insertion time.
  pub created_at: Timestamp,
}

/// A chunk claimed for extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimedChunk {
  /// The claimed row (`status = Extracting`).
  pub chunk: Chunk,
}

/// A fact row about to be written inside [`Store::commit_chunk`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewFact {
  /// Engine-allocated fact id.
  pub id: FactId,
  /// Owning bank.
  pub bank_id: BankId,
  /// Owning network.
  pub fact_type: FactType,
  /// Narrative text.
  pub text: String,
  /// Source chunk; `None` for observations and reflect-formed opinions.
  pub chunk_id: Option<ChunkId>,
  /// Embedding vector; `None` means backfill is pending.
  pub embedding: Option<Vec<f32>>,
  /// Occurrence interval start.
  pub occurred_start: Option<Timestamp>,
  /// Occurrence interval end.
  pub occurred_end: Option<Timestamp>,
  /// Mention time.
  pub mentioned_at: Option<Timestamp>,
  /// Belief strength (required for reflect-formed opinions).
  pub confidence: Option<f32>,
  /// Remaining auxiliary metadata (paper field `x`).
  pub attrs: Option<serde_json::Value>,
  /// Creation time.
  pub created_at: Timestamp,
}

/// A `fact_entity` binding about to be written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewBinding {
  /// Owning bank; must match the fact and the entity.
  pub bank_id: BankId,
  /// Bound fact.
  pub fact_id: FactId,
  /// Bound entity.
  pub entity_id: EntityId,
  /// Raw mention at extraction time.
  pub mention_text: Option<String>,
  /// Creation time.
  pub created_at: Timestamp,
}

/// A `fact_link` edge about to be written.
///
/// Temporal edges are undirected and must be supplied as two reciprocal rows;
/// semantic and causal edges are directed single rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewLink {
  /// Edge identifier.
  pub id: LinkId,
  /// Owning bank; must match both endpoints.
  pub bank_id: BankId,
  /// Initiating side for directed edges.
  pub from: FactId,
  /// Other endpoint.
  pub to: FactId,
  /// Edge type.
  pub link_type: LinkType,
  /// Causal relation; required for causal edges.
  pub relation: Option<CausalRelation>,
  /// Edge weight.
  pub weight: f32,
  /// Creation time.
  pub created_at: Timestamp,
}

/// A delta to an entity's evidence statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityStatUpdate {
  /// Entity to update.
  pub entity_id: EntityId,
  /// `true` adds an evidence row, `false` removes one (re-extraction).
  pub add: bool,
  /// Conversation time of the evidence (drives `first_seen`/`last_seen`);
  /// `None` leaves the seen bounds unchanged.
  pub at: Option<Timestamp>,
  /// When this update is applied.
  pub updated_at: Timestamp,
}

/// A confidence adjustment (and optional rewrite) for an existing opinion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpinionConfidenceUpdate {
  /// Opinion row to update.
  pub opinion_id: FactId,
  /// New belief strength, already clamped by the engine.
  pub new_confidence: f32,
  /// Replacement text when a strong contradiction rewrites the opinion.
  pub new_text: Option<String>,
  /// When this update is applied.
  pub updated_at: Timestamp,
}

/// Everything written by the chunk transaction (tx2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkCommit {
  /// Chunk being completed.
  pub chunk_id: ChunkId,
  /// Claim token: the `updated_at` value written by [`Store::claim_chunk`].
  ///
  /// The status flip is conditional on it, so a worker whose claim was stolen by
  /// a stale reclaim cannot commit (the store returns `Error::Conflict`).
  pub claimed_at: Timestamp,
  /// When the chunk reached `done` (also the chunk's `updated_at`).
  pub completed_at: Timestamp,
  /// Fact rows produced by extraction.
  pub facts: Vec<NewFact>,
  /// New entities produced by resolution. A `UNIQUE(bank_id, canonical_name)`
  /// conflict reuses the existing row, and the store rewrites bindings that
  /// referenced the conflicting candidate id.
  pub entities: Vec<Entity>,
  /// Fact↔entity bindings.
  pub bindings: Vec<NewBinding>,
  /// Graph edges.
  pub links: Vec<NewLink>,
  /// Entity statistics deltas.
  pub entity_stats: Vec<EntityStatUpdate>,
  /// Opinion reinforcements/weakenings.
  pub opinion_updates: Vec<OpinionConfidenceUpdate>,
}

/// An opinion row plus its entity bindings, written atomically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewOpinion {
  /// Engine-allocated fact id.
  pub id: FactId,
  /// Owning bank.
  pub bank_id: BankId,
  /// Opinion statement.
  pub text: String,
  /// Belief strength, already clamped by the engine.
  pub confidence: f32,
  /// Bound entities with optional mention text.
  pub entities: Vec<(EntityId, Option<String>)>,
  /// Embedding vector; `None` means backfill is pending.
  pub embedding: Option<Vec<f32>>,
  /// Creation time.
  pub created_at: Timestamp,
}

/// An observation row plus its entity bindings, appended as a batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewObservation {
  /// Engine-allocated fact id.
  pub fact_id: FactId,
  /// Owning bank.
  pub bank_id: BankId,
  /// Observation statement.
  pub text: String,
  /// Embedding vector; `None` means backfill is pending.
  pub embedding: Option<Vec<f32>>,
  /// Entities this observation describes (multiple bindings are allowed).
  pub entity_ids: Vec<EntityId>,
  /// Creation time.
  pub created_at: Timestamp,
}

/// Work waiting for a background pass; every list is derived in-database.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkLists {
  /// Sources still in `received`.
  pub sources_received: Vec<SourceId>,
  /// Chunks still in `pending`.
  pub chunks_pending: Vec<ChunkId>,
  /// Chunks stuck in `extracting` past the staleness threshold.
  pub chunks_stale: Vec<ChunkId>,
  /// Facts with a `NULL` embedding.
  pub facts_missing_embedding: Vec<FactId>,
  /// Entities whose evidence is newer than their latest observation batch.
  pub entities_stale_observation: Vec<EntityId>,
}

/// Adjacency kind reported by [`Store::neighbors`].
///
/// `Entity` is derived (never stored in `fact_link`) and exists as its own kind
/// because spreading activation applies a distinct multiplier per kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeighborKind {
  /// Temporal proximity (undirected, written as a reciprocal pair).
  Temporal,
  /// Vector affinity (directed: "from selected to").
  Semantic,
  /// Causal relation (directed: cause → effect).
  Causal,
  /// Shared-entity adjacency derived from `fact_entity` (weight always 1.0).
  Entity,
}

/// One outgoing edge of the memory graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Neighbor {
  /// The neighboring fact.
  pub fact_id: FactId,
  /// How the facts are adjacent.
  pub kind: NeighborKind,
  /// Edge weight.
  pub weight: f32,
}

/// An entity candidate for resolution, with the variants observed so far.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityCandidate {
  /// The candidate entity row.
  pub entity: Entity,
  /// Mention variants aggregated from `fact_entity.mention_text`.
  pub variants: Vec<String>,
  /// Most recent mention time across the variants.
  pub last_mention_at: Option<Timestamp>,
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::{BankProfile, ChunkStatus, SourceStatus};

  /// Compile-time proof that the contract is object-safe and injectable.
  struct StubStore;

  #[async_trait]
  impl Store for StubStore {
    async fn create_bank(&self, _bank: Bank) -> Result<()> {
      unreachable!("stub")
    }
    async fn load_bank(&self, id: BankId) -> Result<Bank> {
      Err(crate::Error::not_found("bank", id.to_string()))
    }
    async fn create_source(&self, _source: Source) -> Result<()> {
      unreachable!("stub")
    }
    async fn append_chunks(&self, _source: SourceId, _chunks: Vec<NewChunk>) -> Result<usize> {
      unreachable!("stub")
    }
    async fn claim_chunk(
      &self,
      _bank: BankId,
      _now: Timestamp,
      _stale_after: Duration,
    ) -> Result<Option<ClaimedChunk>> {
      unreachable!("stub")
    }
    async fn release_chunk(&self, _chunk: ChunkId) -> Result<()> {
      unreachable!("stub")
    }
    async fn commit_chunk(&self, _commit: ChunkCommit) -> Result<()> {
      unreachable!("stub")
    }
    async fn search_vector(
      &self,
      _bank: BankId,
      _query: &[f32],
      _k: usize,
    ) -> Result<Vec<(FactId, f32)>> {
      unreachable!("stub")
    }
    async fn search_bm25(
      &self,
      _bank: BankId,
      _query: &str,
      _k: usize,
    ) -> Result<Vec<(FactId, f32)>> {
      unreachable!("stub")
    }
    async fn neighbors(&self, _bank: BankId, _from: FactId) -> Result<Vec<Neighbor>> {
      unreachable!("stub")
    }
    async fn facts_in_range(
      &self,
      _bank: BankId,
      _range: crate::types::TimeRange,
    ) -> Result<Vec<FactId>> {
      unreachable!("stub")
    }
    async fn fetch_facts(&self, _ids: &[FactId]) -> Result<Vec<Fact>> {
      unreachable!("stub")
    }
    async fn increment_access(&self, _ids: &[FactId]) -> Result<()> {
      unreachable!("stub")
    }
    async fn ensure_entity(&self, _new: NewEntity) -> Result<(EntityId, bool)> {
      unreachable!("stub")
    }
    async fn entity_candidates(
      &self,
      _bank: BankId,
      _mention: &str,
    ) -> Result<Vec<EntityCandidate>> {
      unreachable!("stub")
    }
    async fn insert_opinion(&self, _opinion: NewOpinion) -> Result<()> {
      unreachable!("stub")
    }
    async fn insert_observations(&self, _observations: Vec<NewObservation>) -> Result<()> {
      unreachable!("stub")
    }
    async fn observations_of(
      &self,
      _bank: BankId,
      _entity: EntityId,
      _limit: usize,
    ) -> Result<Vec<Fact>> {
      unreachable!("stub")
    }
    async fn pending_work(
      &self,
      _bank: BankId,
      _now: Timestamp,
      _stale_after: Duration,
    ) -> Result<WorkLists> {
      unreachable!("stub")
    }
    async fn update_embeddings(&self, _embeddings: &[(FactId, Vec<f32>)]) -> Result<usize> {
      unreachable!("stub")
    }
    async fn gc_observations(
      &self,
      _bank: BankId,
      _now: Timestamp,
      _max_age: Duration,
      _per_entity_cap: usize,
    ) -> Result<usize> {
      unreachable!("stub")
    }
  }

  const fn assert_send_sync<T: Send + Sync + 'static>() {}

  #[test]
  fn store_is_object_safe_and_send_sync() {
    assert_send_sync::<Box<dyn Store>>();
    let _store: Box<dyn Store> = Box::new(StubStore);
  }

  #[tokio::test]
  async fn dyn_injection_dispatches_through_the_trait_object() {
    let boxed: Box<dyn Store> = Box::new(StubStore);
    let error = boxed.load_bank(BankId::new()).await.expect_err("stub bank");
    assert!(matches!(error, crate::Error::NotFound("bank", _)));

    let shared: std::sync::Arc<dyn Store> = std::sync::Arc::new(StubStore);
    let error = shared
      .load_bank(BankId::new())
      .await
      .expect_err("stub bank");
    assert!(matches!(error, crate::Error::NotFound("bank", _)));
  }

  #[test]
  fn dto_serde_round_trip() {
    let now = Timestamp::from_timestamp_millis(1_700_000_000_000).expect("timestamp");
    let bank_id = BankId::new();
    let fact_id = FactId::new();
    let entity_id = EntityId::new();
    let chunk_id = ChunkId::new();

    let commit = ChunkCommit {
      chunk_id,
      claimed_at: now,
      completed_at: now,
      facts: vec![NewFact {
        id: fact_id,
        bank_id,
        fact_type: FactType::World,
        text: "Emily joined Google.".into(),
        chunk_id: Some(chunk_id),
        embedding: Some(vec![0.1, 0.2]),
        occurred_start: None,
        occurred_end: None,
        mentioned_at: Some(now),
        confidence: None,
        attrs: Some(serde_json::json!({ "fact_kind": "statement" })),
        created_at: now,
      }],
      entities: vec![Entity {
        id: entity_id,
        bank_id,
        canonical_name: "emily".into(),
        entity_type: EntityKind::Person,
        mention_count: 0,
        first_seen: None,
        last_seen: None,
        attrs: None,
        created_at: now,
        updated_at: now,
      }],
      bindings: vec![NewBinding {
        bank_id,
        fact_id,
        entity_id,
        mention_text: Some("Emily".into()),
        created_at: now,
      }],
      links: vec![NewLink {
        id: LinkId::new(),
        bank_id,
        from: fact_id,
        to: FactId::new(),
        link_type: LinkType::Causal,
        relation: Some(CausalRelation::Causes),
        weight: 1.0,
        created_at: now,
      }],
      entity_stats: vec![EntityStatUpdate {
        entity_id,
        add: true,
        at: Some(now),
        updated_at: now,
      }],
      opinion_updates: vec![OpinionConfidenceUpdate {
        opinion_id: FactId::new(),
        new_confidence: 0.7,
        new_text: None,
        updated_at: now,
      }],
    };

    let encoded = serde_json::to_string(&commit).expect("serialize commit");
    let decoded: ChunkCommit = serde_json::from_str(&encoded).expect("deserialize commit");

    assert_eq!(decoded, commit);
  }

  #[test]
  fn neighbor_kinds_use_snake_case() {
    assert_eq!(
      serde_json::to_string(&NeighborKind::Entity).expect("serialize"),
      "\"entity\""
    );
    assert_eq!(
      serde_json::to_string(&NeighborKind::Causal).expect("serialize"),
      "\"causal\""
    );
  }

  #[test]
  fn rows_still_round_trip_through_the_port_dtos() {
    // Guards against accidental field drift between row structs and DTOs.
    let now = Timestamp::from_timestamp_millis(1_700_000_000_000).expect("timestamp");
    let source = Source {
      id: SourceId::new(),
      bank_id: BankId::new(),
      title: Some("transcript".into()),
      payload_text: "hello".into(),
      status: SourceStatus::Received,
      created_at: now,
      updated_at: now,
    };
    assert_eq!(source.status, SourceStatus::Received);

    let chunk = NewChunk {
      id: ChunkId::new(),
      bank_id: source.bank_id,
      source_id: source.id,
      chunk_index: 0,
      text: "hello".into(),
      content_hash: "hash".into(),
      time_start: None,
      time_end: None,
      created_at: now,
    };
    let encoded = serde_json::to_string(&chunk).expect("serialize chunk");
    let decoded: NewChunk = serde_json::from_str(&encoded).expect("deserialize chunk");
    assert_eq!(decoded, chunk);

    // `ClaimedChunk` exposes the row as `pending → extracting` in the adapter.
    let claimed = ClaimedChunk {
      chunk: Chunk {
        id: chunk.id,
        bank_id: chunk.bank_id,
        source_id: chunk.source_id,
        chunk_index: 0,
        text: chunk.text,
        content_hash: chunk.content_hash,
        status: ChunkStatus::Extracting,
        time_start: None,
        time_end: None,
        created_at: now,
        updated_at: now,
      },
    };
    assert_eq!(claimed.chunk.status, ChunkStatus::Extracting);
  }

  #[test]
  fn bank_profile_is_available_to_the_port() {
    let profile = BankProfile {
      background: "I am a careful assistant.".into(),
      skepticism: 3,
      literalism: 3,
      empathy: 3,
      bias_strength: 0.2,
    };
    assert!(profile.validate().is_ok());
  }
}
