//! retain · inbox, chunking, extraction orchestration, tx2 commit, rescan.

pub mod chunker;
pub mod normalize;

pub use chunker::{ChunkConfig, RawChunk, chunk, chunk_hash};
pub use normalize::{NormalizedPayload, RawPayload, RawTurn, TurnSpan, normalize};
