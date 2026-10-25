//! Time-ordered identifiers.
//!
//! Every entity in the system uses a UUIDv7 newtype. UUIDv7 is lexicographically
//! ordered by creation time, which keeps primary keys sortable and pagination
//! stable. IDs are allocated by the engine before a transaction starts, so the
//! store never has to generate business identifiers.

use std::{fmt, str::FromStr};

use uuid::Uuid;

/// Error returned when an ID string cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdParseError(String);

impl fmt::Display for IdParseError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "invalid id: {}", self.0)
  }
}

impl std::error::Error for IdParseError {}

macro_rules! id_type {
  ($name:ident, $doc:literal) => {
    #[doc = $doc]
    #[derive(
      Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
    )]
    #[serde(transparent)]
    pub struct $name(Uuid);

    impl $name {
      /// Creates a new time-ordered (UUIDv7) identifier.
      pub fn new() -> Self {
        Self(Uuid::now_v7())
      }

      /// Parses an identifier from its string form.
      pub fn parse(value: &str) -> Result<Self, IdParseError> {
        Uuid::try_parse(value)
          .map(Self)
          .map_err(|err| IdParseError(err.to_string()))
      }

      /// Returns the underlying UUID.
      pub fn as_uuid(&self) -> Uuid {
        self.0
      }
    }

    impl Default for $name {
      fn default() -> Self {
        Self::new()
      }
    }

    impl fmt::Display for $name {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
      }
    }

    impl FromStr for $name {
      type Err = IdParseError;

      fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
      }
    }
  };
}

id_type!(BankId, "Identifier of a memory bank.");
id_type!(SourceId, "Identifier of an ingested source (ledger entry).");
id_type!(ChunkId, "Identifier of a chunk slice.");
id_type!(FactId, "Identifier of a fact row.");
id_type!(EntityId, "Identifier of a canonical entity.");

#[cfg(test)]
mod tests {
  use std::time::Duration;

  use super::*;

  #[test]
  fn parses_and_displays_canonical_form() {
    let id = BankId::new();
    let text = id.to_string();

    let parsed = BankId::parse(&text).expect("canonical id should parse");
    assert_eq!(parsed, id);

    let via_from_str: BankId = text.parse().expect("FromStr should agree with parse");
    assert_eq!(via_from_str, id);
  }

  #[test]
  fn serde_round_trip_is_transparent() {
    let id = FactId::new();
    let json = serde_json::to_string(&id).expect("serialize");

    assert_eq!(json, format!("\"{id}\""));

    let back: FactId = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, id);
  }

  #[test]
  fn successive_ids_are_time_ordered() {
    let mut previous = EntityId::new();

    for _ in 0..3 {
      std::thread::sleep(Duration::from_millis(2));
      let next = EntityId::new();
      assert!(previous < next, "UUIDv7 ids must increase over time");
      previous = next;
    }
  }

  #[test]
  fn rejects_invalid_input() {
    assert!(ChunkId::parse("not-a-uuid").is_err());
    assert!("not-a-uuid".parse::<SourceId>().is_err());
  }
}
