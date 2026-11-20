//! Cross-encoder reranking of fused recall candidates (paper §4.2.4).
//!
//! RRF produces a stable order from rank evidence only. The reranker scores the
//! query against the top candidates jointly and reorders that prefix; the rest
//! of the list keeps its fused order. The port is best-effort: a failure falls
//! back to the fused order instead of failing recall.

use tracing::warn;

use crate::ports::Reranker;
use crate::types::Fact;

/// Reorders the first `top_n` facts by reranker score.
///
/// The returned vector is the same length and contains the same facts. Only the
/// prefix is reordered, by score descending with ties broken by the original
/// position (so `IdentityReranker` preserves the input). `top_n = 0` disables
/// reranking; a `top_n` larger than the list reranks everything. If the
/// reranker returns the wrong number of scores, or fails, the original order is
/// kept and a warning is logged.
pub async fn rerank_top(
  reranker: &dyn Reranker,
  query: &str,
  facts: &[Fact],
  top_n: usize,
) -> Vec<Fact> {
  let mut ordered: Vec<Fact> = facts.to_vec();
  if top_n == 0 || ordered.len() < 2 {
    return ordered;
  }

  let prefix_len = top_n.min(ordered.len());
  let docs: Vec<&str> = ordered[..prefix_len]
    .iter()
    .map(|fact| fact.text.as_str())
    .collect();

  let scores = match reranker.rerank(query, &docs).await {
    Ok(scores) => scores,
    Err(error) => {
      warn!(%error, "reranker failed; keeping the fused order");
      return ordered;
    }
  };
  if scores.len() != prefix_len {
    warn!(
      expected = prefix_len,
      got = scores.len(),
      "reranker returned the wrong number of scores; keeping the fused order"
    );
    return ordered;
  }

  let mut prefix: Vec<(usize, Fact)> = ordered.drain(..prefix_len).enumerate().collect();
  prefix.sort_by(|left, right| {
    scores[right.0]
      .total_cmp(&scores[left.0])
      .then_with(|| left.0.cmp(&right.0))
  });

  let mut result: Vec<Fact> = prefix.into_iter().map(|(_, fact)| fact).collect();
  result.extend(ordered);
  result
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::{IdentityReranker, timestamp};
  use crate::types::{BankId, FactId, FactType, OccurrenceInterval};
  use crate::{Error, Result};
  use async_trait::async_trait;

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

  struct ReversingReranker;

  #[async_trait]
  impl Reranker for ReversingReranker {
    async fn rerank(&self, _query: &str, docs: &[&str]) -> Result<Vec<f32>> {
      Ok((0..docs.len()).map(|index| index as f32).collect())
    }
  }

  struct FailingReranker;

  #[async_trait]
  impl Reranker for FailingReranker {
    async fn rerank(&self, _query: &str, _docs: &[&str]) -> Result<Vec<f32>> {
      Err(Error::retryable("reranker unavailable"))
    }
  }

  struct WrongLengthReranker;

  #[async_trait]
  impl Reranker for WrongLengthReranker {
    async fn rerank(&self, _query: &str, _docs: &[&str]) -> Result<Vec<f32>> {
      Ok(vec![1.0])
    }
  }

  fn texts(facts: &[Fact]) -> Vec<&str> {
    facts.iter().map(|fact| fact.text.as_str()).collect()
  }

  #[tokio::test]
  async fn identity_keeps_the_fused_order() {
    let facts = vec![fact("a"), fact("b"), fact("c")];
    let reranked = rerank_top(&IdentityReranker, "q", &facts, 3).await;

    assert_eq!(texts(&reranked), vec!["a", "b", "c"]);
  }

  #[tokio::test]
  async fn only_the_prefix_is_reordered() {
    let facts = vec![fact("a"), fact("b"), fact("c"), fact("d"), fact("e")];
    let reranked = rerank_top(&ReversingReranker, "q", &facts, 3).await;

    assert_eq!(
      texts(&reranked),
      vec!["c", "b", "a", "d", "e"],
      "the reranker reverses the first three; the tail is untouched"
    );
  }

  #[tokio::test]
  async fn failure_and_wrong_length_keep_the_original_order() {
    let facts = vec![fact("a"), fact("b"), fact("c")];

    let failed = rerank_top(&FailingReranker, "q", &facts, 3).await;
    assert_eq!(texts(&failed), vec!["a", "b", "c"]);

    let wrong = rerank_top(&WrongLengthReranker, "q", &facts, 3).await;
    assert_eq!(texts(&wrong), vec!["a", "b", "c"]);
  }

  #[tokio::test]
  async fn zero_top_n_and_short_lists_are_no_ops() {
    let facts = vec![fact("a"), fact("b")];

    let disabled = rerank_top(&ReversingReranker, "q", &facts, 0).await;
    assert_eq!(texts(&disabled), vec!["a", "b"]);

    let single = rerank_top(&ReversingReranker, "q", &facts[..1], 5).await;
    assert_eq!(texts(&single), vec!["a"]);

    let empty = rerank_top(&ReversingReranker, "q", &[], 5).await;
    assert!(empty.is_empty());

    let oversized = rerank_top(&ReversingReranker, "q", &facts, 10).await;
    assert_eq!(
      texts(&oversized),
      vec!["b", "a"],
      "top_n larger than the list reranks everything"
    );
  }

  #[tokio::test]
  async fn equal_scores_keep_the_incoming_order() {
    struct Flat;
    #[async_trait]
    impl Reranker for Flat {
      async fn rerank(&self, _query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        Ok(vec![0.5; docs.len()])
      }
    }

    let facts = vec![fact("a"), fact("b"), fact("c")];
    let reranked = rerank_top(&Flat, "q", &facts, 3).await;
    assert_eq!(texts(&reranked), vec!["a", "b", "c"]);
  }
}
