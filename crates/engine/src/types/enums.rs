//! Enumerations shared by the domain model and the persistence layer.
//!
//! All enums serialize as snake_case strings; the store maps them to its own
//! column representation.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The four memory networks from the paper (the `fact_type` column).
///
/// Extraction produces `World`, `Experience`, and `Opinion` rows; `Observation`
/// rows come only from the observation synthesis pipeline (paper A.5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FactType {
  /// Objective facts about the external world.
  World,
  /// The agent's own experiences and actions.
  Experience,
  /// Subjective judgments with confidence scores.
  Opinion,
  /// Preference-neutral entity summaries.
  Observation,
}

/// Edge types materialized in `fact_link`.
///
/// Entity adjacency is deliberately absent: it is derived from `fact_entity`
/// bindings during traversal (schema §0.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LinkType {
  /// Undirected, stored as a reciprocal pair of rows.
  Temporal,
  /// Directed single row: from = the fact that selected the target into its
  /// top-k neighbor budget.
  Semantic,
  /// Directed single row: cause → effect.
  Causal,
}

/// Extraction progress of a chunk (the idempotency and transaction unit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStatus {
  /// Awaiting extraction.
  Pending,
  /// Claimed by a worker; claim validity is judged by the row's `updated_at` age.
  Extracting,
  /// The chunk transaction committed.
  Done,
  /// Dead-letter state set by the runtime retry policy; may be reset manually.
  Failed,
}

/// Relation vocabulary for causal edges (paper A.5.1).
///
/// Direction normalization is still an open decision (`design/design.md` §6);
/// the four paper values are kept until it is settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CausalRelation {
  /// The source fact caused the target fact.
  Causes,
  /// The source fact was caused by the target fact.
  CausedBy,
  /// The source fact enables the target fact.
  Enables,
  /// The source fact prevents the target fact.
  Prevents,
}

/// Canonical entity categories (paper §4.1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
  /// A person.
  Person,
  /// An organization.
  Organization,
  /// A location.
  Location,
  /// A product.
  Product,
  /// A concept.
  Concept,
  /// Anything else.
  Other,
}

/// Ingest progress of a source row (the transactional inbox).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceStatus {
  /// Persisted by tx0, awaiting chunking.
  Received,
  /// Chunk rows were written (or all skipped by hash) in tx1.
  Chunked,
}

#[cfg(test)]
mod tests {
  use super::*;

  fn assert_wire_name<T>(value: T, expected: &str)
  where
    T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
  {
    let json = serde_json::to_string(&value).expect("serialize enum");
    assert_eq!(json, format!("\"{expected}\""));
    let back: T = serde_json::from_str(&json).expect("deserialize enum");
    assert_eq!(back, value);
  }

  #[test]
  fn enums_use_snake_case_wire_names() {
    assert_wire_name(FactType::World, "world");
    assert_wire_name(FactType::Observation, "observation");
    assert_wire_name(LinkType::Semantic, "semantic");
    assert_wire_name(ChunkStatus::Extracting, "extracting");
    assert_wire_name(CausalRelation::CausedBy, "caused_by");
    assert_wire_name(EntityKind::Organization, "organization");
    assert_wire_name(SourceStatus::Chunked, "chunked");
  }
}
