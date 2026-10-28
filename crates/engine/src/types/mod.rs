//! Core domain types shared across the engine.

pub mod enums;
pub mod id;
pub mod profile;
pub mod rows;
pub mod structured;
pub mod time;

pub use enums::{
  CausalRelation, ChunkStatus, EntityKind, EnumParseError, FactType, LinkType, SourceStatus,
};
pub use id::{BankId, ChunkId, EntityId, FactId, IdParseError, LinkId, SourceId};
pub use profile::{BankProfile, ProfileError};
pub use rows::{Bank, Chunk, Entity, Fact, FactEntity, FactLink, Source};
pub use structured::{
  Assessment, AssessmentVerdict, CausalRef, EntityMention, ExtractedFact, FactExtractionResponse,
  ObservationDraft, ObservationResponse, OpinionDraft, ReflectResponse,
};
pub use time::{TimeRange, Timestamp, from_epoch_millis, to_epoch_millis};
