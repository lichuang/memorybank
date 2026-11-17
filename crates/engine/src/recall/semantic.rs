//! Semantic recall channel (paper §4.2.2, eqs. 9–10).
//!
//! The store owns the vector index; this module is the recall-level policy:
//! request the top-k in the target scope and drop matches below the configured
//! cosine floor. It deliberately does not reimplement search or fetch facts.

use crate::ports::{FactScope, Store};
use crate::types::{BankId, FactId};
use crate::{Error, Result};

/// Semantic-channel parameters. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticConfig {
  /// Number of candidates requested from the store. `0` disables the channel.
  pub top_k: usize,
  /// Minimum cosine similarity kept after the store call, in `[-1, 1]`.
  pub min_cos: f32,
}

impl Default for SemanticConfig {
  fn default() -> Self {
    Self {
      top_k: 50,
      min_cos: 0.0,
    }
  }
}

impl SemanticConfig {
  /// Validates the cosine floor. `top_k = 0` is a valid disabled state.
  pub fn validate(&self) -> Result<()> {
    if !self.min_cos.is_finite() || !(-1.0..=1.0).contains(&self.min_cos) {
      return Err(Error::config(
        "semantic min_cos must be finite and in [-1, 1]",
      ));
    }
    Ok(())
  }
}

/// Returns the top semantic matches in `scope`, highest cosine first.
///
/// The store contract already returns at most `top_k` rows in ranked order and
/// skips facts without embeddings; this wrapper applies the configured
/// `min_cos` floor and preserves that order. A `top_k` of zero returns an empty
/// list without calling the store.
pub async fn semantic_channel(
  store: &dyn Store,
  bank: BankId,
  query: &[f32],
  cfg: &SemanticConfig,
  scope: &FactScope,
) -> Result<Vec<(FactId, f32)>> {
  cfg.validate()?;
  if cfg.top_k == 0 {
    return Ok(Vec::new());
  }

  let mut results = store.search_vector(bank, query, cfg.top_k, scope).await?;

  // `NaN >= x` is false, so a corrupt score is dropped as well.
  results.retain(|(_, score)| *score >= cfg.min_cos);
  results.truncate(cfg.top_k);
  Ok(results)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_config_is_valid() {
    assert!(SemanticConfig::default().validate().is_ok());
  }

  #[test]
  fn rejects_an_out_of_range_floor() {
    for min_cos in [-1.1, 1.1, f32::NAN, f32::INFINITY] {
      let config = SemanticConfig {
        min_cos,
        ..SemanticConfig::default()
      };
      assert!(
        matches!(config.validate(), Err(Error::Config(_))),
        "min_cos: {min_cos}"
      );
    }
  }

  #[test]
  fn zero_top_k_is_a_valid_disabled_state() {
    let config = SemanticConfig {
      top_k: 0,
      ..SemanticConfig::default()
    };
    assert!(config.validate().is_ok());
  }
}
