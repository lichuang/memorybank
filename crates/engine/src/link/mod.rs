//! Link construction: temporal, semantic, and causal edges.
//!
//! Thresholds and budgets live here as runtime configuration; none of them is
//! persisted (the schema stores only the edges that were finally kept).

pub mod causal;
pub mod semantic;
pub mod temporal;

use std::time::Duration;

/// Thresholds and budgets shared by the link builders.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkConfig {
  /// Temporal candidate window: two facts further apart than this are never
  /// linked, no matter how small the decayed weight would be.
  pub temporal_window: Duration,
  /// Decay scale: `weight = exp(-gap / temporal_sigma)`.
  pub temporal_sigma: Duration,
  /// Neighbor budget per new fact: each new fact keeps its `temporal_top_k`
  /// closest candidates (pairs are deduplicated before writing).
  pub temporal_top_k: usize,
  /// Neighbor budget per new fact for semantic edges.
  pub semantic_top_k: usize,
  /// Cosine threshold θs: candidates below it never become edges.
  pub semantic_threshold: f32,
}

impl Default for LinkConfig {
  fn default() -> Self {
    Self {
      temporal_window: Duration::from_secs(24 * 60 * 60),
      temporal_sigma: Duration::from_secs(6 * 60 * 60),
      temporal_top_k: 5,
      semantic_top_k: 5,
      semantic_threshold: 0.8,
    }
  }
}

pub use causal::causal_links;
pub use semantic::{build_semantic_links, cosine_similarity};
pub use temporal::{build_temporal_links, temporal_weight};
