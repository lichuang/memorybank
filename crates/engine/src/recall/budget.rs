//! Token-budget packing of the final recall list (paper §4.2.1, eq. 8).
//!
//! The budget is a hard cap on the combined token count, not a per-fact limit:
//! facts are taken greedily in final ranked order and packing stops at the first
//! one that would overflow. A fact that alone exceeds the budget is skipped
//! rather than truncating its text — a truncated fact would be a fabricated
//! quote of generated content.

use tracing::warn;

use crate::tokens::TokenEstimator;
use crate::types::Fact;

/// Packs `facts` into a token budget, preserving their order.
///
/// Returns the kept facts and the tokens consumed. Budget `0` keeps nothing.
/// Because packing stops at the first overflow, a small high-ranked fact is
/// never displaced by a larger lower-ranked one. A fact whose estimate alone
/// exceeds the remaining budget is skipped (logged when it is the first one, the
/// "cannot answer within budget" case).
pub fn pack(
  facts: &[(Fact, f32)],
  budget: usize,
  estimator: &dyn TokenEstimator,
) -> (Vec<Fact>, usize) {
  let mut kept = Vec::new();
  let mut used = 0_usize;

  for (index, (fact, _score)) in facts.iter().enumerate() {
    let cost = estimator.estimate(&fact.text);
    let Some(next) = used.checked_add(cost) else {
      warn!(
        index,
        cost, "token estimate overflowed; stopping the budget"
      );
      break;
    };
    if next > budget {
      if index == 0 {
        warn!(
          cost,
          budget, "the top-ranked fact alone exceeds the token budget; returning empty"
        );
      }
      break;
    }
    used = next;
    kept.push(fact.clone());
  }

  (kept, used)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::timestamp;
  use crate::tokens::CharHeuristic;
  use crate::types::{BankId, FactId, FactType, OccurrenceInterval};

  fn fact(text: &str) -> Fact {
    Fact {
      id: FactId::new(),
      bank_id: BankId::new(),
      fact_type: FactType::World,
      text: text.into(),
      embedding: None,
      chunk_id: None,
      occurred: OccurrenceInterval::Unknown,
      mentioned_at: None,
      confidence: None,
      access_count: 0,
      attrs: None,
      created_at: timestamp(0),
      updated_at: timestamp(0),
    }
  }

  fn ranked(texts: &[&str]) -> Vec<(Fact, f32)> {
    texts
      .iter()
      .enumerate()
      .map(|(index, text)| (fact(text), 1.0 - index as f32 * 0.1))
      .collect()
  }

  fn texts(facts: &[Fact]) -> Vec<&str> {
    facts.iter().map(|fact| fact.text.as_str()).collect()
  }

  /// One token per character keeps the arithmetic obvious.
  fn estimator() -> CharHeuristic {
    CharHeuristic::new(1.0)
  }

  #[test]
  fn packs_until_the_budget_is_exhausted() {
    // Costs are 4 / 4 / 4.
    let facts = ranked(&["aaaa", "bbbb", "cccc"]);

    let (kept, used) = pack(&facts, 8, &estimator());
    assert_eq!(texts(&kept), vec!["aaaa", "bbbb"]);
    assert_eq!(used, 8);

    let (all, used) = pack(&facts, 12, &estimator());
    assert_eq!(texts(&all), vec!["aaaa", "bbbb", "cccc"]);
    assert_eq!(used, 12);

    let (one, used) = pack(&facts, 5, &estimator());
    assert_eq!(texts(&one), vec!["aaaa"]);
    assert_eq!(used, 4);
  }

  #[test]
  fn stops_at_the_first_overflow_instead_of_backfilling() {
    // Costs are 6 / 2 / 2: a 7-token budget keeps only the first fact.
    let facts = ranked(&["aaaaaa", "bb", "cc"]);

    let (kept, used) = pack(&facts, 7, &estimator());

    assert_eq!(
      texts(&kept),
      vec!["aaaaaa"],
      "the lower-ranked small facts do not replace the first overflow"
    );
    assert_eq!(used, 6);
  }

  #[test]
  fn zero_budget_keeps_nothing() {
    let (kept, used) = pack(&ranked(&["a"]), 0, &estimator());
    assert!(kept.is_empty());
    assert_eq!(used, 0);
  }

  #[test]
  fn a_first_fact_over_budget_returns_empty() {
    let (kept, used) = pack(&ranked(&["aaaaaa", "b"]), 3, &estimator());
    assert!(kept.is_empty());
    assert_eq!(used, 0);
  }

  #[test]
  fn empty_input_is_empty() {
    let (kept, used) = pack(&[], 100, &estimator());
    assert!(kept.is_empty());
    assert_eq!(used, 0);
  }

  #[test]
  fn exact_boundary_is_inclusive() {
    let (kept, used) = pack(&ranked(&["abcd"]), 4, &estimator());
    assert_eq!(kept.len(), 1);
    assert_eq!(used, 4);
  }

  #[test]
  fn empty_text_costs_nothing_and_is_kept() {
    let facts = vec![(fact(""), 1.0), (fact("abcd"), 0.9)];

    let (kept, used) = pack(&facts, 4, &estimator());

    assert_eq!(kept.len(), 2, "a zero-cost fact does not consume budget");
    assert_eq!(used, 4);
  }

  #[test]
  fn default_heuristic_uses_quarter_tokens_per_char() {
    let facts = ranked(&["abcdefgh"]); // ceil(8 * 0.25) = 2

    let (kept, used) = pack(&facts, 2, &CharHeuristic::default());

    assert_eq!(kept.len(), 1);
    assert_eq!(used, 2);
  }
}
