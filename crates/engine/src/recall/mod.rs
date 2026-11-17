//! Recall channels. P2-1 starts with the semantic (vector) channel; the
//! lexical, graph, and temporal channels land in later tasks.

pub mod bm25;
pub mod semantic;

pub use bm25::{Bm25Config, bm25_channel};
pub use semantic::{SemanticConfig, semantic_channel};
