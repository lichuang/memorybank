//! retain · inbox, chunking, extraction orchestration, tx2 commit, rescan.

pub mod augment;
pub mod chunker;
pub mod embed;
pub mod extract;
pub mod normalize;

pub use augment::{augmented_text, fact_text, human_readable, normalize_time};
pub use chunker::{ChunkConfig, RawChunk, chunk, chunk_hash};
pub use embed::{TextEmbedding, embed_facts};
pub use extract::{ExtractionConfig, ExtractionReport, extract_facts, validate_extracted};
pub use normalize::{NormalizedPayload, RawPayload, RawTurn, TurnSpan, normalize};
