//! Core domain types shared across the engine.

pub mod enums;
pub mod id;
pub mod profile;
pub mod rows;

pub use enums::{CausalRelation, ChunkStatus, EntityKind, FactType, LinkType, SourceStatus};
pub use id::{BankId, ChunkId, EntityId, FactId, IdParseError, LinkId, SourceId};
pub use profile::{BankProfile, ProfileError};
pub use rows::{Bank, Chunk, Entity, Fact, FactEntity, FactLink, Source};

/// A UTC timestamp. All wall-clock values in the system are UTC.
pub type Timestamp = chrono::DateTime<chrono::Utc>;
