//! Reciprocal Rank Fusion (paper §4.2.3, eq. 15).
//!
//! Fusion is rank-based on purpose: the four channels produce scores on
//! different scales (cosine, BM25, activation, proximity), so raw scores are not
//! comparable. A candidate's fused score is `Σ 1/(k + rank_i)` over the lists it
//! appears in; missing from a list contributes nothing rather than a penalty.

use std::collections::HashMap;

use crate::types::FactId;

/// Default RRF constant from the paper (`k = 60`).
pub const DEFAULT_RRF_K: u32 = 60;

/// Fuses ranked lists into one ranking.
///
/// Ranks are 1-based and positional (the first entry of a list has rank 1).
/// Channels already return unique ids; if a duplicate slips through, its repeat
/// contributes nothing while later entries keep their positional rank, so the
/// fusion never counts one list twice for a single fact. The result
/// contains every id that appears in at least one list, sorted by fused score
/// descending and then id ascending, so equal scores are deterministic. An empty
/// input, or a `k` that would overflow on addition, still produces a stable
/// order rather than panicking.
pub fn rrf(lists: &[Vec<FactId>], k: u32) -> Vec<(FactId, f32)> {
  let mut scores: HashMap<FactId, f32> = HashMap::new();
  let mut seen_in_list: Vec<FactId> = Vec::new();

  for list in lists {
    seen_in_list.clear();
    for (index, id) in list.iter().enumerate() {
      // 1-based rank; skip repeats inside the same list.
      if seen_in_list.contains(id) {
        continue;
      }
      seen_in_list.push(*id);

      let rank = u32::try_from(index + 1).unwrap_or(u32::MAX);
      let contribution = 1.0 / (u64::from(k) + u64::from(rank)) as f32;
      *scores.entry(*id).or_insert(0.0) += contribution;
    }
  }

  let mut fused: Vec<(FactId, f32)> = scores.into_iter().collect();
  fused.sort_by(|left, right| {
    right
      .1
      .total_cmp(&left.1)
      .then_with(|| left.0.cmp(&right.0))
  });
  fused
}

#[cfg(test)]
mod tests {
  use super::*;

  fn id() -> FactId {
    FactId::new()
  }

  #[test]
  fn a_single_list_preserves_its_order() {
    let ids = [id(), id(), id()];
    let fused = rrf(&[ids.to_vec()], DEFAULT_RRF_K);

    assert_eq!(
      fused.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
      ids.to_vec()
    );
    assert!((fused[0].1 - 1.0 / 61.0).abs() < 1e-9);
    assert!((fused[1].1 - 1.0 / 62.0).abs() < 1e-9);
    assert!((fused[2].1 - 1.0 / 63.0).abs() < 1e-9);
  }

  #[test]
  fn crossing_lists_accumulate_multi_channel_support() {
    let shared = id();
    let top = id();
    let second = id();

    // `shared` is second in both lists; `top`/`second` lead one list each.
    let fused = rrf(
      &[vec![top, shared, second], vec![second, shared, top]],
      DEFAULT_RRF_K,
    );
    let scores: HashMap<FactId, f32> = fused.into_iter().collect();

    let shared_score = 2.0 / 62.0;
    assert!((scores[&shared] - shared_score).abs() < 1e-9);
    assert!((scores[&top] - (1.0 / 61.0 + 1.0 / 63.0)).abs() < 1e-9);
    assert!((scores[&second] - (1.0 / 63.0 + 1.0 / 61.0)).abs() < 1e-9);

    let mut expected = vec![
      (shared, shared_score),
      (top, 1.0 / 61.0 + 1.0 / 63.0),
      (second, 1.0 / 61.0 + 1.0 / 63.0),
    ];
    expected.sort_by(|left, right| {
      right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
    });
    assert_eq!(fused_sorted(&scores), expected);
  }

  fn fused_sorted(scores: &HashMap<FactId, f32>) -> Vec<(FactId, f32)> {
    let mut values: Vec<(FactId, f32)> = scores.iter().map(|(id, s)| (*id, *s)).collect();
    values.sort_by(|left, right| {
      right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
    });
    values
  }

  #[test]
  fn a_missing_list_contributes_nothing() {
    let only_in_one = id();
    let fused = rrf(&[vec![only_in_one], vec![]], DEFAULT_RRF_K);

    assert_eq!(fused.len(), 1);
    assert!((fused[0].1 - 1.0 / 61.0).abs() < 1e-9);
  }

  #[test]
  fn ties_break_by_fact_id() {
    let mut ids = [id(), id(), id()];
    ids.sort();

    // Each id is rank 1 in its own list, so all three scores are equal.
    let tied = rrf(&[vec![ids[2]], vec![ids[1]], vec![ids[0]]], DEFAULT_RRF_K);

    assert_eq!(
      tied.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
      ids.to_vec(),
      "equal scores are ordered by ascending fact id"
    );
  }

  #[test]
  fn duplicate_ids_inside_a_list_count_once() {
    let repeated = id();
    let other = id();
    let fused = rrf(&[vec![repeated, repeated, other]], DEFAULT_RRF_K);
    let scores: HashMap<FactId, f32> = fused.into_iter().collect();

    assert!((scores[&repeated] - 1.0 / 61.0).abs() < 1e-9);
    assert!(
      (scores[&other] - 1.0 / 63.0).abs() < 1e-9,
      "rank is positional: the repeat at index 1 still occupies rank 2"
    );
  }

  #[test]
  fn empty_input_is_empty_and_huge_k_stays_finite() {
    assert!(rrf(&[], DEFAULT_RRF_K).is_empty());

    let id = id();
    let fused = rrf(&[vec![id]], u32::MAX);
    assert_eq!(fused.len(), 1);
    assert!(fused[0].1.is_finite());
  }
}
