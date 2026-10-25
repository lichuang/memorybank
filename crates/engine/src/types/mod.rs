//! Core domain types shared across the engine.

pub mod id;

pub use id::{BankId, ChunkId, EntityId, FactId, IdParseError, SourceId};

/// A UTC timestamp. All wall-clock values in the system are UTC.
pub type Timestamp = chrono::DateTime<chrono::Utc>;
