//! Embedding port.

use async_trait::async_trait;

use crate::Result;

/// Embedding provider.
///
/// Write-time and query-time embeddings must come from the same active model;
/// the engine never compares vectors across models.
#[async_trait]
pub trait Embedder: Send + Sync {
  /// Active model identifier.
  fn model(&self) -> &str;

  /// Vector dimension; vectors of any other length are rejected on write.
  fn dim(&self) -> usize;

  /// Embeds a batch, returning one vector per input in order.
  async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

#[cfg(test)]
mod tests {
  use super::*;

  struct StubEmbedder;

  #[async_trait]
  impl Embedder for StubEmbedder {
    fn model(&self) -> &str {
      "stub-embedder"
    }

    fn dim(&self) -> usize {
      2
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
      Ok(texts.iter().map(|_| vec![0.0, 1.0]).collect())
    }
  }

  #[tokio::test]
  async fn dyn_embedder_is_injectable() {
    let embedder: Box<dyn Embedder> = Box::new(StubEmbedder);

    let vectors = embedder.embed(&["a", "b"]).await.expect("embed");

    assert_eq!(embedder.model(), "stub-embedder");
    assert_eq!(embedder.dim(), 2);
    assert_eq!(vectors, vec![vec![0.0, 1.0], vec![0.0, 1.0]]);
  }
}
