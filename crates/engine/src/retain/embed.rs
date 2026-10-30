//! Batch embedding with dimension validation and failure degradation.
//!
//! Embedding happens **before** tx2: a provider outage must not lose extracted
//! facts, so a failed batch is marked [`TextEmbedding::Failed`] and the row is
//! picked up later from `facts_missing_embedding`. A dimension mismatch, by
//! contrast, is a configuration error — writing the vector would poison vector
//! search, so the whole call fails.

use tracing::warn;

use crate::ports::Embedder;
use crate::{Error, Result};

/// The embedding outcome for one input text.
#[derive(Debug, Clone, PartialEq)]
pub enum TextEmbedding {
  /// The provider returned a vector for this text.
  Vector(Vec<f32>),
  /// The provider failed for the whole batch containing this text. The fact is
  /// still committed without an embedding and is backfilled later.
  Failed,
}

impl TextEmbedding {
  /// Converts to the storage shape: [`None`] means "backfill later".
  pub fn into_option(self) -> Option<Vec<f32>> {
    match self {
      Self::Vector(vector) => Some(vector),
      Self::Failed => None,
    }
  }
}

/// Embeds `texts` in batches, preserving input order.
///
/// Returns one [`TextEmbedding`] per input. Provider failures mark the affected
/// batch [`TextEmbedding::Failed`] (logged) so tx2 can still commit: `Retryable`
/// outages are backfilled later, and `Fatal` degradation preserves the
/// extracted facts instead of losing them over an embedding. A provider
/// [`Error::Config`] is returned immediately — silently emitting null
/// embeddings would hide a broken setup. A returned vector whose length does
/// not match [`Embedder::dim`] is [`Error::Config`], and a provider returning a
/// different number of vectors than inputs is [`Error::Fatal`] because the
/// outputs can no longer be aligned with the texts.
///
/// `batch == 0` is rejected as a configuration error; an empty input performs
/// no provider call.
pub async fn embed_facts(
  embedder: &dyn Embedder,
  texts: &[String],
  batch: usize,
) -> Result<Vec<TextEmbedding>> {
  if batch == 0 {
    return Err(Error::config("embedding batch size must be positive"));
  }
  if texts.is_empty() {
    return Ok(Vec::new());
  }

  let dim = embedder.dim();
  let mut embeddings = Vec::with_capacity(texts.len());

  for (index, group) in texts.chunks(batch).enumerate() {
    let batch_texts: Vec<&str> = group.iter().map(String::as_str).collect();

    match embedder.embed(&batch_texts).await {
      Ok(vectors) => {
        if vectors.len() != group.len() {
          return Err(Error::fatal(format!(
            "embedder returned {} vectors for {} inputs",
            vectors.len(),
            group.len()
          )));
        }

        for vector in vectors {
          if vector.len() != dim {
            return Err(Error::config(format!(
              "embedder returned dimension {} but the model declares {dim}",
              vector.len()
            )));
          }
          embeddings.push(TextEmbedding::Vector(vector));
        }
      }
      Err(error @ Error::Config(_)) => return Err(error),
      Err(error) => {
        warn!(
          batch = index,
          rows = group.len(),
          %error,
          "embedding batch failed; leaving rows unembedded for backfill"
        );
        embeddings.extend(std::iter::repeat_n(TextEmbedding::Failed, group.len()));
      }
    }
  }

  Ok(embeddings)
}

#[cfg(test)]
mod tests {
  use std::sync::Mutex;

  use async_trait::async_trait;

  use super::*;

  /// Deterministic test double that records batch sizes and can inject faults.
  struct TestEmbedder {
    dim: usize,
    returned_dim: usize,
    returned_count: Option<usize>,
    failure: Option<(usize, Error)>,
    calls: Mutex<Vec<usize>>,
  }

  impl TestEmbedder {
    fn new(dim: usize) -> Self {
      Self {
        dim,
        returned_dim: dim,
        returned_count: None,
        failure: None,
        calls: Mutex::new(Vec::new()),
      }
    }

    fn failing(dim: usize, call: usize) -> Self {
      Self::failing_with(dim, call, Error::retryable("scripted provider failure"))
    }

    fn failing_with(dim: usize, call: usize, error: Error) -> Self {
      Self {
        failure: Some((call, error)),
        ..Self::new(dim)
      }
    }

    fn mismatched_dim(dim: usize, returned_dim: usize) -> Self {
      Self {
        returned_dim,
        ..Self::new(dim)
      }
    }

    fn wrong_count(dim: usize, returned_count: usize) -> Self {
      Self {
        returned_count: Some(returned_count),
        ..Self::new(dim)
      }
    }

    fn calls(&self) -> Vec<usize> {
      self.calls.lock().expect("calls lock").clone()
    }
  }

  #[async_trait]
  impl Embedder for TestEmbedder {
    fn model(&self) -> &str {
      "test-embedder"
    }

    fn dim(&self) -> usize {
      self.dim
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
      let call = {
        let mut calls = self.calls.lock().expect("calls lock");
        calls.push(texts.len());
        calls.len() - 1
      };

      if let Some((failing_call, error)) = &self.failure
        && *failing_call == call
      {
        return Err(error.clone());
      }

      let count = self.returned_count.unwrap_or(texts.len());
      let payload: Vec<Vec<f32>> = texts
        .iter()
        .map(|text| vec![text.chars().count() as f32; self.returned_dim])
        .collect();
      Ok(payload.into_iter().take(count).collect())
    }
  }

  fn texts(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
  }

  fn vector(embedding: &TextEmbedding) -> &[f32] {
    match embedding {
      TextEmbedding::Vector(vector) => vector,
      TextEmbedding::Failed => panic!("expected a vector, got a failed batch"),
    }
  }

  #[test]
  fn converts_to_the_storage_shape() {
    assert_eq!(
      TextEmbedding::Vector(vec![0.5, 1.0]).into_option(),
      Some(vec![0.5, 1.0])
    );
    assert_eq!(TextEmbedding::Failed.into_option(), None);
  }

  #[tokio::test]
  async fn splits_into_batches_and_preserves_order() {
    let embedder = TestEmbedder::new(2);
    let values = texts(&["a", "bb", "ccc", "dddd", "eeeee"]);

    let embeddings = embed_facts(&embedder, &values, 2).await.expect("embed");

    assert_eq!(embedder.calls(), vec![2, 2, 1]);
    assert_eq!(embeddings.len(), values.len());
    assert!(
      embeddings
        .iter()
        .all(|embedding| matches!(embedding, TextEmbedding::Vector(_)))
    );
    assert!(
      embeddings
        .iter()
        .all(|embedding| vector(embedding).len() == 2)
    );

    // The first component encodes the text length, so order is observable.
    let first_components: Vec<f32> = embeddings
      .iter()
      .map(|embedding| vector(embedding)[0])
      .collect();
    assert_eq!(first_components, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
  }

  #[tokio::test]
  async fn failed_batch_is_marked_and_later_batches_continue() {
    let embedder = TestEmbedder::failing(1, 1);
    let values = texts(&["a", "bb", "ccc", "dddd", "eeeee"]);

    let embeddings = embed_facts(&embedder, &values, 2)
      .await
      .expect("failure is degraded, not returned");

    assert_eq!(embedder.calls(), vec![2, 2, 1]);
    assert!(matches!(embeddings[0], TextEmbedding::Vector(_)));
    assert!(matches!(embeddings[1], TextEmbedding::Vector(_)));
    assert!(
      matches!(embeddings[2], TextEmbedding::Failed),
      "the failing batch is marked"
    );
    assert!(matches!(embeddings[3], TextEmbedding::Failed));
    assert!(
      matches!(embeddings[4], TextEmbedding::Vector(_)),
      "later batches still run"
    );
  }

  #[tokio::test]
  async fn fatal_provider_error_also_degrades_the_batch() {
    let embedder = TestEmbedder::failing_with(1, 0, Error::fatal("malformed response"));
    let values = texts(&["a", "bb", "ccc"]);

    let embeddings = embed_facts(&embedder, &values, 2)
      .await
      .expect("a fatal provider error still preserves facts");

    assert!(matches!(embeddings[0], TextEmbedding::Failed));
    assert!(matches!(embeddings[1], TextEmbedding::Failed));
    assert!(matches!(embeddings[2], TextEmbedding::Vector(_)));
  }

  #[tokio::test]
  async fn config_provider_error_is_returned_and_stops_further_batches() {
    let embedder = TestEmbedder::failing_with(2, 0, Error::config("missing api key"));
    let values = texts(&["a", "bb", "ccc", "dddd"]);

    let error = embed_facts(&embedder, &values, 2)
      .await
      .expect_err("config errors must not be hidden");

    assert!(matches!(error, Error::Config(_)), "error: {error}");
    assert_eq!(embedder.calls(), vec![2], "later batches are not attempted");
  }

  #[tokio::test]
  async fn dimension_mismatch_is_a_config_error() {
    let embedder = TestEmbedder::mismatched_dim(3, 2);

    let error = embed_facts(&embedder, &texts(&["a"]), 4)
      .await
      .expect_err("dimension mismatch");

    assert!(matches!(error, Error::Config(_)), "error: {error}");
  }

  #[tokio::test]
  async fn wrong_vector_count_is_fatal() {
    let embedder = TestEmbedder::wrong_count(2, 1);

    let error = embed_facts(&embedder, &texts(&["a", "b"]), 2)
      .await
      .expect_err("vector count mismatch");

    assert!(matches!(error, Error::Fatal(_)), "error: {error}");
  }

  #[tokio::test]
  async fn rejects_a_zero_batch_size() {
    let embedder = TestEmbedder::new(2);

    let error = embed_facts(&embedder, &texts(&["a"]), 0)
      .await
      .expect_err("zero batch");

    assert!(matches!(error, Error::Config(_)), "error: {error}");
    assert!(embedder.calls().is_empty());
  }

  #[tokio::test]
  async fn empty_input_does_not_call_the_provider() {
    let embedder = TestEmbedder::new(2);

    let embeddings = embed_facts(&embedder, &[], 8).await.expect("embed");

    assert!(embeddings.is_empty());
    assert!(embedder.calls().is_empty());
  }
}
