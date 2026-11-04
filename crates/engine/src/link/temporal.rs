//! Temporal link construction.
//!
//! Two facts are temporally related when their occurrence intervals are close:
//! the gap is zero for overlapping or touching intervals and otherwise the
//! empty span between them. The edge weight decays as `exp(-gap / sigma)`, and
//! because the predicate is symmetric every kept pair is written as two rows.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::ports::{NewLink, Store};
use crate::types::{
  BankId, Fact, FactId, LinkId, LinkType, TimeRange, Timestamp, from_epoch_millis, to_epoch_millis,
};
use crate::{Error, Result};

use super::LinkConfig;

/// Exponential temporal decay: `exp(-gap / sigma)`.
///
/// Paper §4.1.4 ("Link Types and Graph Structure", 1) Temporal Links),
/// eq. (4): `w_ij^temp = exp(-Δt_ij / σt)`, where Δt_ij is the time difference
/// between two memories and σt is the decay parameter. This project measures
/// Δt as the gap between occurrence intervals (0 when they overlap or touch).
///
/// A zero `sigma` degrades to a step function (only zero-gap pairs link) instead
/// of dividing by zero; [`LinkConfig`] defaults keep it positive.
pub fn temporal_weight(gap: Duration, sigma: Duration) -> f32 {
  if sigma.is_zero() {
    return if gap.is_zero() { 1.0 } else { 0.0 };
  }

  (-(gap.as_secs_f64()) / sigma.as_secs_f64()).exp() as f32
}

/// Builds temporal links between `new_facts` and the bank's existing facts.
///
/// Temporal edges encode "these two memories happened close in time". The
/// predicate is symmetric, so every kept pair is returned as **two** rows
/// (`from → to` and `to → from`) with the same weight; a one-sided temporal
/// edge is never produced.
///
/// # What is compared
///
/// Only the occurrence interval (`occurred_start..occurred_end`) is used, not
/// the mention time. A fact must have **both** bounds to participate: an open
/// interval has no meaningful gap and is skipped. Intervals with `end < start`
/// are normalized before scoring.
///
/// # Algorithm
///
/// 1. **Filter + normalize**: keep the new facts that have a complete interval.
///    When none do, return an empty vector without touching the store.
/// 2. **Candidate windows**: expand each interval by
///    [`LinkConfig::temporal_window`] on both sides and merge overlapping
///    windows, so a scattered batch does not scan the same rows once per fact.
/// 3. **Candidate rows**: call [`Store::facts_in_range`] for each merged window
///    (occurrence intersection; `occurred_end IS NULL` is an open interval),
///    deduplicate the ids and load them in one [`Store::fetch_facts`] call.
/// 4. **Score**: for every new fact, compute the gap to each existing fact and
///    to the other new facts. `gap == 0` for overlapping or touching intervals;
///    otherwise it is the empty span between them. Pairs beyond the window are
///    dropped, and the rest get `exp(-gap / sigma)`.
/// 5. **Budget**: each new fact keeps its [`LinkConfig::temporal_top_k`]
///    strongest candidates (weight first, id as a deterministic tie-break),
///    then the selected pairs are deduplicated by the canonical key
///    `(min(from, to), max(from, to))`.
/// 6. **Emit**: each surviving pair becomes two [`NewLink`] rows with
///    `link_type = Temporal`, `relation = None`, the same weight, and
///    `created_at = now`.
///
/// Existing-existing pairs are not considered: they were already built when
/// those facts were committed. Only pairs with at least one endpoint in
/// `new_facts` can be produced, which also keeps the write amplification bounded
/// by the size of the current extraction pass.
///
/// # Inputs
///
/// - `new_facts`: facts produced by the current extraction pass. They live in
///   memory and are not yet visible to the store, so new-new pairs are matched
///   here rather than in SQL.
/// - `config`: runtime thresholds; nothing in it is persisted.
/// - `now`: the only clock use, stamping the edge rows. The weight comes from
///   the facts' event times, never from wall-clock time.
///
/// # Errors
///
/// Store read failures propagate. An interval expansion that leaves the
/// representable timestamp range is [`Error::Config`].
///
/// # Guarantees
///
/// - the output is deterministic for the same inputs (link ids are freshly
///   allocated, but the selected pairs and weights are stable);
/// - every pair is reciprocal (`from → to` and `to → from`, equal weight), so
///   the database invariant "canonical-key group count is 2" can never be
///   broken by this builder;
/// - at most `2 * temporal_top_k * new_facts.len()` rows are returned;
/// - facts from other banks never appear (the range query is bank-scoped).
pub async fn build_temporal_links(
  store: &dyn Store,
  bank: BankId,
  new_facts: &[Fact],
  config: &LinkConfig,
  now: Timestamp,
) -> Result<Vec<NewLink>> {
  // Only facts with both occurrence bounds can produce a meaningful gap; open
  // intervals are skipped instead of being linked at gap zero.
  let timed: Vec<(&Fact, Interval)> = new_facts
    .iter()
    .filter_map(|fact| interval(fact).map(|interval| (fact, interval)))
    .collect();
  if timed.is_empty() {
    return Ok(Vec::new());
  }

  let window_ms = millis(config.temporal_window);
  let mut expanded: Vec<(i64, i64)> = timed
    .iter()
    .map(|(_, interval)| {
      (
        interval.start.saturating_sub(window_ms),
        interval.end.saturating_add(window_ms),
      )
    })
    .collect();
  expanded.sort_unstable();

  // Merge overlapping windows so a scattered batch does not scan the same rows
  // once per new fact.
  let mut ranges: Vec<(i64, i64)> = Vec::new();
  for (start, end) in expanded {
    if let Some(last) = ranges.last_mut()
      && start <= last.1
    {
      last.1 = last.1.max(end);
    } else {
      ranges.push((start, end));
    }
  }

  // New facts are not committed yet, so the store normally cannot return them;
  // filtering keeps the function correct if a caller also passes persisted
  // facts as `new_facts`.
  let new_ids: HashSet<FactId> = new_facts.iter().map(|fact| fact.id).collect();
  let mut existing_ids: HashSet<FactId> = HashSet::new();
  for (start, end) in ranges {
    let range = TimeRange {
      start: from_epoch_millis(start)
        .ok_or_else(|| Error::config("temporal window start is out of range"))?,
      end: from_epoch_millis(end)
        .ok_or_else(|| Error::config("temporal window end is out of range"))?,
    };
    for id in store.facts_in_range(bank, range).await? {
      if !new_ids.contains(&id) {
        existing_ids.insert(id);
      }
    }
  }

  let mut ids: Vec<FactId> = existing_ids.into_iter().collect();
  ids.sort_unstable();
  let existing = if ids.is_empty() {
    Vec::new()
  } else {
    store.fetch_facts(&ids).await?
  };

  let mut selected: HashMap<(FactId, FactId), f32> = HashMap::new();
  for (fact, fact_interval) in &timed {
    let mut candidates: Vec<(FactId, f32)> = Vec::new();

    // Score against both sides: stored facts (new→existing) and the rest of the
    // batch (new→new). Existing-existing pairs were linked when those facts
    // were committed, so they are intentionally not re-scored here.

    for other in &existing {
      if let Some(other_interval) = interval(other)
        && let Some(weight) = pair_weight(*fact_interval, other_interval, config)
      {
        candidates.push((other.id, weight));
      }
    }
    for (other, other_interval) in &timed {
      if other.id == fact.id {
        continue;
      }
      if let Some(weight) = pair_weight(*fact_interval, *other_interval, config) {
        candidates.push((other.id, weight));
      }
    }

    // Deterministic ordering: strongest first, ids break ties (UUIDv7 order).
    candidates.sort_by(|left, right| {
      right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
    });
    // Per-fact budget: a dense time cluster cannot consume the whole commit's
    // edge allowance.
    candidates.truncate(config.temporal_top_k);

    for (other, weight) in candidates {
      // The pair may be selected from both sides; canonicalize so each
      // unordered pair is emitted once.
      let key = canonical_pair(fact.id, other);
      selected.entry(key).or_insert(weight);
    }
  }

  let mut pairs: Vec<(FactId, FactId, f32)> = selected
    .into_iter()
    .map(|((from, to), weight)| (from, to, weight))
    .collect();
  pairs.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

  // Reciprocal rows are emitted together: the schema considers a temporal pair
  // complete only when both directions exist with the same weight.
  let mut links = Vec::with_capacity(pairs.len() * 2);
  for (from, to, weight) in pairs {
    links.push(NewLink {
      id: LinkId::new(),
      bank_id: bank,
      from,
      to,
      link_type: LinkType::Temporal,
      relation: None,
      weight,
      created_at: now,
    });
    links.push(NewLink {
      id: LinkId::new(),
      bank_id: bank,
      from: to,
      to: from,
      link_type: LinkType::Temporal,
      relation: None,
      weight,
      created_at: now,
    });
  }

  Ok(links)
}

/// A fact's occurrence interval, normalized so `start <= end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interval {
  start: i64,
  end: i64,
}

/// Returns `None` unless both occurrence bounds are present.
fn interval(fact: &Fact) -> Option<Interval> {
  let start = fact.occurred_start.map(to_epoch_millis)?;
  let end = fact.occurred_end.map(to_epoch_millis)?;

  Some(if end < start {
    Interval {
      start: end,
      end: start,
    }
  } else {
    Interval { start, end }
  })
}

/// Empty span between two closed intervals; zero when they overlap or touch.
fn gap_ms(left: Interval, right: Interval) -> i64 {
  if left.end < right.start {
    right.start.saturating_sub(left.end)
  } else if right.end < left.start {
    left.start.saturating_sub(right.end)
  } else {
    0
  }
}

/// Scores one candidate pair, or returns `None` when it falls outside the
/// window.
///
/// The gap is symmetric and clamped at zero for overlapping or touching
/// intervals. The window check is inclusive (`gap == window` still links).
/// Pruning returns `None` rather than a zero weight, so "not eligible" stays
/// distinct from "eligible but decayed to ~0". The returned weight is
/// `exp(-gap / sigma)` (paper §4.1.4 eq. 4), always in `[0, 1]`.
fn pair_weight(left: Interval, right: Interval, config: &LinkConfig) -> Option<f32> {
  let gap = gap_ms(left, right);
  if gap > millis(config.temporal_window) {
    return None;
  }

  // `gap_ms` is non-negative by construction, so the cast is lossless.
  Some(temporal_weight(
    Duration::from_millis(gap as u64),
    config.temporal_sigma,
  ))
}

/// Orders a fact pair by id so both selection directions collapse to one key.
///
/// Temporal edges are undirected and the same pair can be selected from either
/// side (A picks B, B picks A). Canonicalizing to `(min, max)` lets the builder
/// deduplicate those hits and emit the pair once. `FactId` is UUIDv7, so the
/// order is stable and roughly creation-ordered.
fn canonical_pair(left: FactId, right: FactId) -> (FactId, FactId) {
  if left <= right {
    (left, right)
  } else {
    (right, left)
  }
}

fn millis(duration: Duration) -> i64 {
  i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::FactType;

  fn fact(start: Option<i64>, end: Option<i64>) -> Fact {
    Fact {
      id: FactId::new(),
      bank_id: BankId::new(),
      fact_type: FactType::World,
      text: "fact".into(),
      embedding: None,
      chunk_id: None,
      occurred_start: start.map(crate::testing::timestamp),
      occurred_end: end.map(crate::testing::timestamp),
      mentioned_at: None,
      confidence: None,
      access_count: 0,
      attrs: None,
      created_at: crate::testing::timestamp(0),
      updated_at: crate::testing::timestamp(0),
    }
  }

  fn config(window_ms: u64, sigma_ms: u64) -> LinkConfig {
    LinkConfig {
      temporal_window: Duration::from_millis(window_ms),
      temporal_sigma: Duration::from_millis(sigma_ms),
      temporal_top_k: 5,
    }
  }

  #[test]
  fn weight_decays_exponentially() {
    let sigma = Duration::from_secs(3_600);

    assert_eq!(temporal_weight(Duration::ZERO, sigma), 1.0);

    let one_sigma = temporal_weight(Duration::from_secs(3_600), sigma);
    assert!((one_sigma - (-1.0_f32).exp()).abs() < 1e-6, "{one_sigma}");

    let two_sigma = temporal_weight(Duration::from_secs(7_200), sigma);
    assert!((two_sigma - (-2.0_f32).exp()).abs() < 1e-6, "{two_sigma}");

    assert!(two_sigma < one_sigma);
  }

  #[test]
  fn zero_sigma_degrades_to_a_step() {
    assert_eq!(temporal_weight(Duration::ZERO, Duration::ZERO), 1.0);
    assert_eq!(
      temporal_weight(Duration::from_millis(1), Duration::ZERO),
      0.0
    );
  }

  #[test]
  fn gap_is_zero_for_overlapping_or_touching_intervals() {
    let left = Interval {
      start: 0,
      end: 1_000,
    };

    assert_eq!(
      gap_ms(
        left,
        Interval {
          start: 500,
          end: 1_500
        }
      ),
      0
    );
    assert_eq!(
      gap_ms(
        left,
        Interval {
          start: 1_000,
          end: 2_000
        }
      ),
      0,
      "touching intervals count as adjacent"
    );
  }

  #[test]
  fn gap_measures_the_empty_span() {
    let left = Interval {
      start: 0,
      end: 1_000,
    };

    assert_eq!(
      gap_ms(
        left,
        Interval {
          start: 4_000,
          end: 5_000
        }
      ),
      3_000
    );
    assert_eq!(
      gap_ms(
        Interval {
          start: 4_000,
          end: 5_000
        },
        left
      ),
      3_000,
      "the gap is symmetric"
    );
  }

  #[test]
  fn interval_requires_both_bounds_and_normalizes_order() {
    assert!(interval(&fact(Some(1_000), None)).is_none());
    assert!(interval(&fact(None, Some(1_000))).is_none());
    assert!(interval(&fact(None, None)).is_none());

    let reversed = interval(&fact(Some(2_000), Some(1_000))).expect("both bounds");
    assert_eq!(
      reversed,
      Interval {
        start: 1_000,
        end: 2_000
      }
    );
  }

  #[test]
  fn pair_weight_enforces_the_window() {
    let window = 4 * 3_600_000;
    let link_config = config(window, 3_600_000);
    let origin = Interval { start: 0, end: 0 };
    let boundary = Interval {
      start: window as i64,
      end: window as i64,
    };
    let beyond = Interval {
      start: window as i64 + 1,
      end: window as i64 + 1,
    };

    let weight = pair_weight(origin, boundary, &link_config).expect("gap == window is inclusive");
    assert!((weight - (-4.0_f32).exp()).abs() < 1e-6, "weight: {weight}");

    assert!(
      pair_weight(origin, beyond, &link_config).is_none(),
      "gap > window is pruned"
    );

    let zero_sigma = config(window, 0);
    assert_eq!(pair_weight(origin, origin, &zero_sigma), Some(1.0));
    assert_eq!(
      pair_weight(origin, Interval { start: 1, end: 1 }, &zero_sigma),
      Some(0.0),
      "a zero sigma only keeps touching intervals"
    );
  }

  #[test]
  fn canonical_pair_is_order_independent() {
    let first = FactId::new();
    let second = FactId::new();
    let expected = if first <= second {
      (first, second)
    } else {
      (second, first)
    };

    assert_eq!(canonical_pair(first, second), expected);
    assert_eq!(canonical_pair(second, first), expected);
    assert!(expected.0 <= expected.1);
  }

  #[test]
  fn millis_saturates_for_huge_durations() {
    assert_eq!(millis(Duration::ZERO), 0);
    assert_eq!(millis(Duration::from_millis(1_234)), 1_234);
    assert_eq!(millis(Duration::MAX), i64::MAX);
  }
}
