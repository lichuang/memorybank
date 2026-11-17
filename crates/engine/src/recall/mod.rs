//! Recall channels. P2-1 starts with the semantic (vector) channel; the
//! lexical, graph, and temporal channels land in later tasks.

pub mod semantic;

pub use semantic::{SemanticConfig, semantic_channel};
