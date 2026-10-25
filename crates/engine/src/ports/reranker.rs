//! Reranking port.

use async_trait::async_trait;

use crate::Result;

/// Cross-encoder reranker used to blend or sanity-check retrieval scores.
#[async_trait]
pub trait Reranker: Send + Sync {
  /// Scores each document against `query`; the result is aligned with `docs`
  /// (same length, same order).
  async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;
}

#[cfg(test)]
mod tests {
  use super::*;

  struct StubReranker;

  #[async_trait]
  impl Reranker for StubReranker {
    async fn rerank(&self, _query: &str, docs: &[&str]) -> Result<Vec<f32>> {
      Ok(docs.iter().map(|_| 0.5).collect())
    }
  }

  #[tokio::test]
  async fn dyn_reranker_keeps_one_score_per_document() {
    let reranker: Box<dyn Reranker> = Box::new(StubReranker);

    let scores = reranker
      .rerank("who is Emily?", &["a", "b"])
      .await
      .expect("rerank");

    assert_eq!(scores.len(), 2);
  }
}
