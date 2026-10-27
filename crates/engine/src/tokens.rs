//! Token estimation shared by chunking (P1-2) and budget packing (P2-7).

/// Estimates the token count of a text.
///
/// The engine never hard-codes a tokenizer: adapters and tests inject whatever
/// estimator the deployment needs (character heuristic first, a real tokenizer
/// later).
pub trait TokenEstimator: Send + Sync {
  /// Returns the estimated token count of `text`.
  fn estimate(&self, text: &str) -> usize;
}

/// A character-count heuristic: `ceil(chars × ratio)`.
///
/// `ratio` is tokens per character. Reasonable starting points: `0.25` for
/// English (~4 chars/token) and `1.0` for CJK. It is a sizing heuristic, not a
/// billing-accurate count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharHeuristic {
  ratio: f32,
}

impl CharHeuristic {
  /// Creates a heuristic with `ratio` tokens per character.
  ///
  /// # Panics
  ///
  /// Panics when `ratio` is not positive and finite.
  pub fn new(ratio: f32) -> Self {
    assert!(
      ratio.is_finite() && ratio > 0.0,
      "token ratio must be positive and finite"
    );
    Self { ratio }
  }

  /// Returns the configured tokens-per-character ratio.
  pub fn ratio(&self) -> f32 {
    self.ratio
  }
}

impl Default for CharHeuristic {
  fn default() -> Self {
    Self::new(0.25)
  }
}

impl TokenEstimator for CharHeuristic {
  fn estimate(&self, text: &str) -> usize {
    let chars = text.chars().count();
    (chars as f32 * self.ratio).ceil() as usize
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_heuristic_uses_quarter_token_per_char() {
    let estimator = CharHeuristic::default();

    assert_eq!(estimator.ratio(), 0.25);
    assert_eq!(estimator.estimate(""), 0);
    assert_eq!(estimator.estimate("abcd"), 1);
    assert_eq!(estimator.estimate("abcde"), 2);
  }

  #[test]
  fn cjk_ratio_counts_one_token_per_char() {
    let estimator = CharHeuristic::new(1.0);

    assert_eq!(estimator.estimate("你好世界"), 4);
  }
}
