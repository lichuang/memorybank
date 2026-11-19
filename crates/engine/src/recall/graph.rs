//! Graph recall channel: spreading activation (paper §4.2.2, eq. 12).
//!
//! Seeds are the semantic hits. Activation propagates over `Store::neighbors`
//! (typed `fact_link` edges plus derived entity adjacency) using
//! `A(fj) = max(A(fi) * w * delta * mu(kind))`. The result includes the seeds
//! and is sorted by activation (descending, id tie-break).

use std::collections::{HashMap, VecDeque};

use crate::ports::{FactScope, NeighborKind, Store};
use crate::types::{BankId, FactId};
use crate::{Error, Result};

/// Spreading-activation parameters. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphConfig {
  /// Decay factor δ in `(0, 1]`.
  pub delta: f32,
  /// Per-kind multipliers µ(ℓ).
  pub mu_temporal: f32,
  /// Per-kind multipliers µ(ℓ).
  pub mu_semantic: f32,
  /// Per-kind multipliers µ(ℓ).
  pub mu_entity: f32,
  /// Per-kind multipliers µ(ℓ).
  pub mu_causal: f32,
  /// Maximum number of edges from a seed (`0` returns seeds only).
  pub max_depth: u8,
  /// Maximum number of expanded nodes, including seeds.
  pub max_visits: usize,
}

impl Default for GraphConfig {
  fn default() -> Self {
    Self {
      delta: 0.5,
      mu_temporal: 0.5,
      mu_semantic: 0.8,
      mu_entity: 1.2,
      mu_causal: 1.5,
      max_depth: 2,
      max_visits: 1_000,
    }
  }
}

impl GraphConfig {
  /// Validates δ and the µ multipliers. `max_depth = 0` is valid (seeds only).
  pub fn validate(&self) -> Result<()> {
    if !self.delta.is_finite() || self.delta <= 0.0 || self.delta > 1.0 {
      return Err(Error::config("graph delta must be finite in (0, 1]"));
    }
    for (name, value) in [
      ("mu_temporal", self.mu_temporal),
      ("mu_semantic", self.mu_semantic),
      ("mu_entity", self.mu_entity),
      ("mu_causal", self.mu_causal),
    ] {
      if !value.is_finite() || value < 0.0 {
        return Err(Error::config(format!(
          "graph {name} must be finite and non-negative"
        )));
      }
    }
    if self.max_visits == 0 {
      return Err(Error::config("graph max_visits must be positive"));
    }
    Ok(())
  }
}

/// Runs spreading activation from `seeds`.
///
/// Non-finite and non-positive seed scores are ignored; duplicate seeds keep the
/// highest score. The output contains every activated fact, including seeds,
/// sorted by score descending and fact id ascending. `max_depth` counts edges
/// from a seed; `max_visits` limits how many queued nodes are expanded.
pub async fn spread(
  store: &dyn Store,
  bank: BankId,
  seeds: &[(FactId, f32)],
  cfg: &GraphConfig,
) -> Result<Vec<(FactId, f32)>> {
  cfg.validate()?;

  let mut activation: HashMap<FactId, f32> = HashMap::new();
  for (id, score) in seeds {
    if !score.is_finite() || *score <= 0.0 {
      continue;
    }
    activation
      .entry(*id)
      .and_modify(|current| *current = current.max(*score))
      .or_insert(*score);
  }
  if activation.is_empty() {
    return Ok(Vec::new());
  }

  // Deterministic starting order; the heap/queue is not exposed.
  let mut initial: Vec<FactId> = activation.keys().copied().collect();
  initial.sort();
  let mut queue: VecDeque<(FactId, u8)> = initial.into_iter().map(|id| (id, 0)).collect();

  let mut visits = 0_usize;
  while let Some((node, depth)) = queue.pop_front() {
    if visits >= cfg.max_visits {
      break;
    }
    visits += 1;
    if depth >= cfg.max_depth {
      continue;
    }

    let base = activation.get(&node).copied().unwrap_or(0.0);
    for neighbor in store.neighbors(bank, node, &FactScope::All).await? {
      let mu = match neighbor.kind {
        NeighborKind::Temporal => cfg.mu_temporal,
        NeighborKind::Semantic => cfg.mu_semantic,
        NeighborKind::Entity => cfg.mu_entity,
        NeighborKind::Causal => cfg.mu_causal,
      };
      let candidate = base * neighbor.weight * cfg.delta * mu;
      if !candidate.is_finite() || candidate <= 0.0 {
        continue;
      }

      let current = activation.get(&neighbor.fact_id).copied().unwrap_or(0.0);
      if candidate > current {
        activation.insert(neighbor.fact_id, candidate);
        queue.push_back((neighbor.fact_id, depth + 1));
      }
    }
  }

  let mut results: Vec<(FactId, f32)> = activation.into_iter().collect();
  results.sort_by(|left, right| {
    right
      .1
      .total_cmp(&left.1)
      .then_with(|| left.0.cmp(&right.0))
  });
  Ok(results)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_config_is_valid() {
    assert!(GraphConfig::default().validate().is_ok());
  }

  #[test]
  fn invalid_config_is_rejected() {
    for config in [
      GraphConfig {
        delta: 0.0,
        ..GraphConfig::default()
      },
      GraphConfig {
        delta: 1.1,
        ..GraphConfig::default()
      },
      GraphConfig {
        delta: f32::NAN,
        ..GraphConfig::default()
      },
      GraphConfig {
        mu_entity: -0.1,
        ..GraphConfig::default()
      },
      GraphConfig {
        mu_causal: f32::NAN,
        ..GraphConfig::default()
      },
      GraphConfig {
        max_visits: 0,
        ..GraphConfig::default()
      },
    ] {
      assert!(
        matches!(config.validate(), Err(Error::Config(_))),
        "config: {config:?}"
      );
    }
  }
}
