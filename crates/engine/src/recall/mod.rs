//! Recall channels. P2-1 starts with the semantic (vector) channel; the
//! lexical, graph, and temporal channels land in later tasks.

pub mod bm25;
pub mod graph;
pub mod semantic;
pub mod temporal;

pub use bm25::{Bm25Config, bm25_channel};
pub use graph::{GraphConfig, spread};
pub use semantic::{SemanticConfig, semantic_channel};
pub use temporal::{TemporalConfig, temporal_channel};
