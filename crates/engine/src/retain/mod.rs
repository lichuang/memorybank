//! retain · inbox, chunking, extraction orchestration, tx2 commit, rescan.

pub mod chunker;
pub mod extract;
pub mod normalize;

pub use chunker::{ChunkConfig, RawChunk, chunk, chunk_hash};
pub use extract::{ExtractionConfig, ExtractionReport, extract_facts, validate_extracted};
pub use normalize::{NormalizedPayload, RawPayload, RawTurn, TurnSpan, normalize};
