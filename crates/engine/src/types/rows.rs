//! Row structs mirroring the logical schema (`design/memory-schema.md` §3).
//!
//! Every row carries `bank_id` directly — it is the deletion and partition key
//! (schema §0.10). Timestamps are UTC.

use serde::{Deserialize, Serialize};

use super::{
  BankId, BankProfile, CausalRelation, ChunkId, ChunkStatus, EntityId, EntityKind, FactId,
  FactType, LinkId, LinkType, SourceId, SourceStatus, Timestamp,
};

/// A memory bank: the tenant container plus the agent profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bank {
  /// Bank identifier.
  pub id: BankId,
  /// Container name n (unique).
  pub name: String,
  /// Agent profile P = (n, Θ, h).
  pub profile: BankProfile,
  /// Creation time.
  pub created_at: Timestamp,
  /// Last update time.
  pub updated_at: Timestamp,
}

/// An ingest ledger entry (the transactional inbox).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
  /// Source identifier.
  pub id: SourceId,
  /// Owning bank.
  pub bank_id: BankId,
  /// Optional human-readable title.
  pub title: Option<String>,
  /// Canonical full text; chunking is a lossless split of it.
  pub payload_text: String,
  /// Ingest progress.
  pub status: SourceStatus,
  /// Receive time (inbox entry time).
  pub created_at: Timestamp,
  /// Last update time.
  pub updated_at: Timestamp,
}

/// A lossless slice of a source payload (idempotency and transaction unit).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
  /// Chunk identifier.
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
  /// Extraction progress.
  pub status: ChunkStatus,
  /// Persistence time (tx1).
  pub created_at: Timestamp,
  /// Conversation-clock interval covered by the chunk.
  pub time_start: Option<Timestamp>,
  /// Conversation-clock interval covered by the chunk.
  pub time_end: Option<Timestamp>,
  /// Last update time; refreshed on claim and status flips.
  pub updated_at: Timestamp,
}

/// A fact row — the paper's memory unit across all four fact types.
///
/// Invariants (schema §3.4):
/// - `fact_type == Observation` implies `chunk_id == None`;
/// - reflect-produced opinions have `chunk_id == None` and `confidence.is_some()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
  /// Fact identifier.
  pub id: FactId,
  /// Owning bank.
  pub bank_id: BankId,
  /// Owning network.
  pub fact_type: FactType,
  /// Narrative fact text (LLM rewrite).
  pub text: String,
  /// Embedding vector; `None` means pending generation/backfill.
  pub embedding: Option<Vec<f32>>,
  /// Chunk slice this row was extracted from; `None` for observations and
  /// reflect-produced opinions.
  pub chunk_id: Option<ChunkId>,
  /// Occurrence interval start τs.
  pub occurred_start: Option<Timestamp>,
  /// Occurrence interval end τe.
  pub occurred_end: Option<Timestamp>,
  /// Mention time τm.
  pub mentioned_at: Option<Timestamp>,
  /// Belief strength c (required for reflect-produced opinions).
  pub confidence: Option<f32>,
  /// Recall hit counter.
  pub access_count: u32,
  /// Remaining auxiliary metadata (paper field `x`): fact_kind, structured
  /// location, and other fields the paper leaves open.
  pub attrs: Option<serde_json::Value>,
  /// Creation time.
  pub created_at: Timestamp,
  /// Last update time.
  pub updated_at: Timestamp,
}

/// A canonical entity produced by entity resolution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
  /// Entity identifier.
  pub id: EntityId,
  /// Owning bank.
  pub bank_id: BankId,
  /// Canonical (lowercased/trimmed) name; unique per bank.
  pub canonical_name: String,
  /// Entity category.
  pub entity_type: EntityKind,
  /// Number of world/experience bindings (evidence-level counter).
  pub mention_count: u32,
  /// First time an evidence row mentioned the entity (conversation clock).
  pub first_seen: Option<Timestamp>,
  /// Most recent such mention.
  pub last_seen: Option<Timestamp>,
  /// Resolution process info (merge history, scores, etc.).
  pub attrs: Option<serde_json::Value>,
  /// Creation time.
  pub created_at: Timestamp,
  /// Last update time.
  pub updated_at: Timestamp,
}

/// A fact↔entity binding; also the physical source of entity adjacency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactEntity {
  /// Owning bank; must match both endpoints.
  pub bank_id: BankId,
  /// Bound fact.
  pub fact_id: FactId,
  /// Bound entity.
  pub entity_id: EntityId,
  /// Raw mention at extraction time, if any.
  pub mention_text: Option<String>,
  /// Creation time.
  pub created_at: Timestamp,
  /// Last update time.
  pub updated_at: Timestamp,
}

/// An edge of the memory graph (temporal, semantic, or causal).
///
/// Entity adjacency is never materialized here; it is derived from
/// `fact_entity` bindings during traversal (schema §3.7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactLink {
  /// Edge identifier.
  pub id: LinkId,
  /// Owning bank; must match both endpoint facts.
  pub bank_id: BankId,
  /// Directed edges: the initiating side (selection or cause).
  pub from_fact_id: FactId,
  /// The other endpoint.
  pub to_fact_id: FactId,
  /// Edge type.
  pub link_type: LinkType,
  /// Causal relation; required for causal edges only.
  pub relation: Option<CausalRelation>,
  /// Edge weight (temporal decay, cosine, or 1.0 for causal).
  pub weight: f32,
  /// Creation time.
  pub created_at: Timestamp,
  /// Last update time.
  pub updated_at: Timestamp,
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn fact_serde_round_trip() {
    let now = chrono::Utc::now();
    let fact = Fact {
      id: FactId::new(),
      bank_id: BankId::new(),
      fact_type: FactType::World,
      text: "Emily works at Google.".into(),
      embedding: Some(vec![0.1, 0.2, 0.3]),
      chunk_id: Some(ChunkId::new()),
      occurred_start: None,
      occurred_end: None,
      mentioned_at: Some(now),
      confidence: None,
      access_count: 2,
      attrs: Some(json!({ "fact_kind": "statement" })),
      created_at: now,
      updated_at: now,
    };

    let encoded = serde_json::to_string(&fact).expect("serialize fact");
    let decoded: Fact = serde_json::from_str(&encoded).expect("deserialize fact");

    assert_eq!(decoded, fact);
  }
}
