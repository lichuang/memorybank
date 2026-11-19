//! Temporal recall channel (paper §4.2.2, eqs. 13–14).
//!
//! A query is parsed into a closed range; the store returns facts whose
//! occurrence interval intersects it; each hit is scored by temporal proximity
//! to the range midpoint. No recognizable time expression means no activation.

use crate::ports::{FactScope, Store, TemporalParser};
use crate::types::{BankId, FactId, Timestamp, to_epoch_millis};
use crate::{Error, Result};

/// Temporal-channel parameters. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalConfig {
  /// Number of candidates kept after scoring. `0` disables the channel.
  pub top_k: usize,
  /// Minimum proximity score kept, in `[0, 1]`.
  pub min_score: f32,
}

impl Default for TemporalConfig {
  fn default() -> Self {
    Self {
      top_k: 50,
      min_score: 0.0,
    }
  }
}

impl TemporalConfig {
  /// Validates the score floor. `top_k = 0` is a valid disabled state.
  pub fn validate(&self) -> Result<()> {
    if !self.min_score.is_finite() || !(0.0..=1.0).contains(&self.min_score) {
      return Err(Error::config(
        "temporal min_score must be finite and in [0, 1]",
      ));
    }
    Ok(())
  }
}

/// Returns temporally relevant facts, closest to the query midpoint first.
///
/// Returns empty when the parser finds no time expression (the channel only
/// activates for temporal queries) or when `top_k` is zero. Facts whose
/// occurrence is unknown cannot be scored and are dropped; open intervals use
/// their start as the midpoint because the event has no known end.
///
/// A zero-width query range (e.g. an exact instant) would make `Δτ/2` zero; the
/// denominator is floored at one millisecond so scores stay finite.
pub async fn temporal_channel(
  store: &dyn Store,
  bank: BankId,
  query: &str,
  now: Timestamp,
  parser: &dyn TemporalParser,
  cfg: &TemporalConfig,
  scope: &FactScope,
) -> Result<Vec<(FactId, f32)>> {
  cfg.validate()?;
  if cfg.top_k == 0 {
    return Ok(Vec::new());
  }

  let Some(range) = parser.parse(query, now) else {
    return Ok(Vec::new());
  };
  if range.end < range.start {
    return Err(Error::config("temporal parser returned a reversed range"));
  }

  let ids = store.facts_in_range(bank, range, scope).await?;
  if ids.is_empty() {
    return Ok(Vec::new());
  }
  let facts = store.fetch_facts(&ids).await?;

  let query_mid = midpoint(range.start, range.end);
  let half_width = (to_epoch_millis(range.end) - to_epoch_millis(range.start)).unsigned_abs() / 2;
  let half_width = half_width.max(1) as f64;

  let mut results: Vec<(FactId, f32)> = facts
    .into_iter()
    .filter_map(|fact| {
      let fact_mid = fact_midpoint(fact.occurred.start(), fact.occurred.end())?;
      let distance = (to_epoch_millis(fact_mid) - to_epoch_millis(query_mid)).unsigned_abs() as f64;
      let score = (1.0 - distance / half_width).clamp(0.0, 1.0) as f32;
      (score >= cfg.min_score).then_some((fact.id, score))
    })
    .collect();

  results.sort_by(|left, right| {
    right
      .1
      .total_cmp(&left.1)
      .then_with(|| left.0.cmp(&right.0))
  });
  results.truncate(cfg.top_k);
  Ok(results)
}

fn midpoint(start: Timestamp, end: Timestamp) -> Timestamp {
  let start = to_epoch_millis(start);
  let end = to_epoch_millis(end);
  let middle = start + (end - start) / 2;
  crate::types::from_epoch_millis(middle)
    .unwrap_or_else(|| crate::types::from_epoch_millis(start).expect("start is valid"))
}

/// The fact's comparison point: the closed midpoint, or the start of an open
/// interval. `None` when the occurrence is unknown.
fn fact_midpoint(start: Option<Timestamp>, end: Option<Timestamp>) -> Option<Timestamp> {
  match (start, end) {
    (Some(start), Some(end)) => Some(midpoint(start, end)),
    (Some(start), None) => Some(start),
    _ => None,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_config_is_valid() {
    assert!(TemporalConfig::default().validate().is_ok());
  }

  #[test]
  fn rejects_an_out_of_range_score_floor() {
    for score in [-0.1, 1.1, f32::NAN] {
      let config = TemporalConfig {
        min_score: score,
        ..TemporalConfig::default()
      };
      assert!(
        matches!(config.validate(), Err(Error::Config(_))),
        "min_score: {score}"
      );
    }
  }

  #[test]
  fn zero_top_k_is_a_valid_disabled_state() {
    let config = TemporalConfig {
      top_k: 0,
      ..TemporalConfig::default()
    };
    assert!(config.validate().is_ok());
  }
}
