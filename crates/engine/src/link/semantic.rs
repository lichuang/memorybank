//! Semantic link construction.
//!
//! Each new fact selects its top-k most similar facts by cosine similarity and
//! emits one **directed** row per selected candidate (`from = selecting fact`).
//! Reciprocity is never assumed: a `B → A` row exists only if B also selects A
//! into its own budget.

use crate::ports::{FactScope, NewLink, Store};
use crate::types::{BankId, Fact, LinkId, LinkType, Timestamp};
use crate::{Error, Result};

use super::LinkConfig;

/// Cosine similarity, the semantic-edge weight (paper §4.1.4, eq. (5)).
///
/// Accumulates in `f64` and returns `0.0` when either vector has zero norm (the
/// angle is undefined). Both slices must have the same length; callers validate
/// that beforehand.
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
  debug_assert_eq!(
    left.len(),
    right.len(),
    "cosine similarity requires equal-length vectors"
  );

  let mut dot = 0.0_f64;
  let mut left_norm = 0.0_f64;
  let mut right_norm = 0.0_f64;

  for (left, right) in left.iter().zip(right) {
    let left = f64::from(*left);
    let right = f64::from(*right);
    dot += left * right;
    left_norm += left * left;
    right_norm += right * right;
  }

  if left_norm == 0.0 || right_norm == 0.0 {
    return 0.0;
  }

  (dot / (left_norm.sqrt() * right_norm.sqrt())) as f32
}

/// Builds semantic links for `new_facts`.
///
/// # Candidate sources
///
/// For every new fact with an embedding:
///
/// 1. [`Store::search_vector`] returns the bank's most similar committed facts
///    (up to `semantic_top_k + 1`, leaving room for a defensive self-filter).
/// 2. The other new facts of the same batch are compared in memory: they are not
///    committed yet, so the store cannot return them, but they must still be
///    able to link to each other.
///
/// Candidates are deduplicated by id, filtered by `semantic_threshold`, sorted by
/// score (ties broken by id, for determinism), and truncated to
/// `semantic_top_k`.
///
/// # Direction
///
/// One row is emitted per selected candidate: `from = new fact`, `to =
/// candidate`, `link_type = Semantic`, `relation = None`, `weight = cosine`
/// clamped to `[0, 1]`. No reverse row is invented. If another new fact selects
/// this one in its own turn, that produces an independent `candidate → new` row.
///
/// # Missing data
///
/// A new fact without an embedding produces no semantic edges; the row is
/// committed without them and backfill/link maintenance can add edges later.
/// A dimension mismatch between vectors is [`Error::Config`] (it means the bank
/// is mixing generations or the caller used the wrong model).
///
/// `semantic_threshold` must be in `[0, 1]`; `semantic_top_k == 0` disables the
/// builder. `now` stamps the edge rows.
pub async fn build_semantic_links(
  store: &dyn Store,
  bank: BankId,
  new_facts: &[Fact],
  config: &LinkConfig,
  now: Timestamp,
) -> Result<Vec<NewLink>> {
  if config.semantic_top_k == 0 {
    return Ok(Vec::new());
  }
  if !(0.0..=1.0).contains(&config.semantic_threshold) {
    return Err(Error::config(format!(
      "semantic threshold must be in [0, 1], got {}",
      config.semantic_threshold
    )));
  }

  let embedded: Vec<(&Fact, &[f32])> = new_facts
    .iter()
    .filter_map(|fact| fact.embedding.as_deref().map(|embedding| (fact, embedding)))
    .collect();
  if embedded.is_empty() {
    return Ok(Vec::new());
  }

  let mut links = Vec::new();

  for (fact, embedding) in &embedded {
    // `+1` keeps the budget intact if this fact is already committed and shows
    // up in its own search results (normally impossible before tx2).
    let mut candidates = store
      .search_vector(
        bank,
        embedding,
        config.semantic_top_k.saturating_add(1),
        &FactScope::All,
      )
      .await?;

    for (other, other_embedding) in &embedded {
      if other.id == fact.id {
        continue;
      }
      if other_embedding.len() != embedding.len() {
        return Err(Error::config(format!(
          "embedding dimension mismatch within the batch: {} vs {}",
          embedding.len(),
          other_embedding.len()
        )));
      }
      candidates.push((other.id, cosine_similarity(embedding, other_embedding)));
    }

    candidates.retain(|(id, score)| *id != fact.id && *score >= config.semantic_threshold);
    candidates.sort_by(|left, right| {
      right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
    });
    // Sorted by descending score first, so dedup keeps the best score per id.
    candidates.dedup_by_key(|(id, _)| *id);
    candidates.truncate(config.semantic_top_k);

    for (to, weight) in candidates {
      links.push(NewLink {
        id: LinkId::new(),
        bank_id: bank,
        from: fact.id,
        to,
        link_type: LinkType::Semantic,
        relation: None,
        weight: weight.clamp(0.0, 1.0),
        created_at: now,
      });
    }
  }

  Ok(links)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cosine_is_one_for_identical_and_scaled_vectors() {
    let vector = [1.0_f32, 2.0, 3.0];
    assert!((cosine_similarity(&vector, &vector) - 1.0).abs() < 1e-6);

    let scaled = [2.0_f32, 4.0, 6.0];
    assert!((cosine_similarity(&vector, &scaled) - 1.0).abs() < 1e-6);
  }

  #[test]
  fn cosine_is_zero_for_orthogonal_vectors() {
    assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
  }

  #[test]
  fn cosine_is_negative_for_opposite_vectors() {
    assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
  }

  #[test]
  fn cosine_is_zero_when_either_norm_is_zero() {
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
  }
}
