//! Enumerations shared by the domain model and the persistence layer.
//!
//! All enums serialize as snake_case strings; the store maps them to its own
//! column representation. [`std::fmt::Display`] / [`std::str::FromStr`] expose
//! the same lowercase wire form so adapters do not need their own string matches.

use std::fmt;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Error returned when a string does not name a valid enum value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumParseError {
  type_name: &'static str,
  value: String,
}

impl EnumParseError {
  fn new(type_name: &'static str, value: impl Into<String>) -> Self {
    Self {
      type_name,
      value: value.into(),
    }
  }

  /// Name of the enum type that rejected the value (e.g. `"fact_type"`).
  pub fn type_name(&self) -> &'static str {
    self.type_name
  }

  /// The rejected value, exactly as supplied.
  pub fn value(&self) -> &str {
    &self.value
  }
}

impl fmt::Display for EnumParseError {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(formatter, "invalid {} `{}`", self.type_name, self.value)
  }
}

impl std::error::Error for EnumParseError {}

/// Implements `as_str`, `Display`, and `FromStr` for a lowercase wire enum.
macro_rules! wire_enum {
  ($name:ident, $label:literal, { $($variant:ident => $text:literal),+ $(,)? }) => {
    impl $name {
      /// Returns the lowercase wire representation (storage column / JSON value).
      pub fn as_str(&self) -> &'static str {
        match self {
          $(Self::$variant => $text),+
        }
      }
    }

    impl fmt::Display for $name {
      fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
      }
    }

    impl FromStr for $name {
      type Err = EnumParseError;

      fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
          $($text => Ok(Self::$variant),)+
          other => Err(EnumParseError::new($label, other)),
        }
      }
    }
  };
}

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

wire_enum!(FactType, "fact_type", {
  World => "world",
  Experience => "experience",
  Opinion => "opinion",
  Observation => "observation",
});

wire_enum!(LinkType, "link_type", {
  Temporal => "temporal",
  Semantic => "semantic",
  Causal => "causal",
});

wire_enum!(ChunkStatus, "chunk_status", {
  Pending => "pending",
  Extracting => "extracting",
  Done => "done",
  Failed => "failed",
});

wire_enum!(CausalRelation, "causal_relation", {
  Causes => "causes",
  CausedBy => "caused_by",
  Enables => "enables",
  Prevents => "prevents",
});

wire_enum!(EntityKind, "entity_kind", {
  Person => "person",
  Organization => "organization",
  Location => "location",
  Product => "product",
  Concept => "concept",
  Other => "other",
});

wire_enum!(SourceStatus, "source_status", {
  Received => "received",
  Chunked => "chunked",
});

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

  fn assert_round_trip<T>(value: T, expected: &str)
  where
    T: Copy + PartialEq + std::fmt::Debug + std::fmt::Display + FromStr,
    T::Err: std::fmt::Debug,
  {
    assert_eq!(value.to_string(), expected);
    let parsed: T = expected.parse().expect("parse wire value");
    assert_eq!(parsed, value);
  }

  #[test]
  fn enums_round_trip_through_display_and_from_str() {
    assert_round_trip(FactType::World, "world");
    assert_round_trip(FactType::Experience, "experience");
    assert_round_trip(FactType::Opinion, "opinion");
    assert_round_trip(FactType::Observation, "observation");
    assert_round_trip(LinkType::Temporal, "temporal");
    assert_round_trip(LinkType::Semantic, "semantic");
    assert_round_trip(LinkType::Causal, "causal");
    assert_round_trip(ChunkStatus::Pending, "pending");
    assert_round_trip(ChunkStatus::Extracting, "extracting");
    assert_round_trip(ChunkStatus::Done, "done");
    assert_round_trip(ChunkStatus::Failed, "failed");
    assert_round_trip(CausalRelation::Causes, "causes");
    assert_round_trip(CausalRelation::CausedBy, "caused_by");
    assert_round_trip(CausalRelation::Enables, "enables");
    assert_round_trip(CausalRelation::Prevents, "prevents");
    assert_round_trip(EntityKind::Person, "person");
    assert_round_trip(EntityKind::Organization, "organization");
    assert_round_trip(EntityKind::Location, "location");
    assert_round_trip(EntityKind::Product, "product");
    assert_round_trip(EntityKind::Concept, "concept");
    assert_round_trip(EntityKind::Other, "other");
    assert_round_trip(SourceStatus::Received, "received");
    assert_round_trip(SourceStatus::Chunked, "chunked");
  }

  #[test]
  fn parsing_rejects_unknown_and_case_mismatched_values() {
    let error = "nonsense".parse::<FactType>().expect_err("invalid value");
    assert_eq!(error.type_name(), "fact_type");
    assert_eq!(error.value(), "nonsense");
    assert_eq!(error.to_string(), "invalid fact_type `nonsense`");

    assert!("World".parse::<FactType>().is_err(), "case-sensitive");
    assert!("".parse::<ChunkStatus>().is_err());
  }
}
