//! Recall: the four retrieval channels plus rank fusion.

pub mod bm25;
pub mod budget;
pub mod graph;
pub mod rerank;
pub mod rrf;
pub mod semantic;
pub mod temporal;

pub use bm25::{Bm25Config, bm25_channel};
pub use budget::pack;
pub use graph::{GraphConfig, spread};
pub use rerank::rerank_top;
pub use rrf::{DEFAULT_RRF_K, rrf};
pub use semantic::{SemanticConfig, semantic_channel};
pub use temporal::{TemporalConfig, temporal_channel};
