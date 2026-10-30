//! Entity resolution ρ (paper §4.1.3).
//!
//! A mention is mapped to a canonical entity by maximizing
//! `α·sim_str + β·sim_co + γ·sim_temp`, with an exact-name fast path and
//! `ensure_entity` as the race-safe creation fallback.

use std::time::Duration;

use crate::ports::{EntityCandidate, NewEntity, Store};
use crate::types::{BankId, EntityId, EntityKind, EntityMention, Timestamp};
use crate::{Error, Result};

/// Resolution weights and thresholds. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveWeights {
  /// Weight of string similarity (always available).
  pub alpha: f32,
  /// Weight of entity co-occurrence (available when the context has entities).
  pub beta: f32,
  /// Weight of temporal proximity (available when both times are known).
  pub gamma: f32,
  /// Minimum weighted score to accept a fuzzy match.
  pub threshold: f32,
  /// Decay scale for temporal proximity: `exp(-Δt / temporal_scale)`.
  pub temporal_scale: Duration,
  /// Maximum number of candidate entities scored per mention.
  pub candidate_limit: usize,
}

impl Default for ResolveWeights {
  fn default() -> Self {
    Self {
      alpha: 0.6,
      beta: 0.2,
      gamma: 0.2,
      threshold: 0.75,
      temporal_scale: Duration::from_secs(30 * 24 * 60 * 60),
      candidate_limit: 500,
    }
  }
}

/// What is already known about the mention being resolved.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MentionContext {
  /// Entities already resolved in the same extraction batch.
  pub co_mentioned: Vec<EntityId>,
  /// Conversation time of the mention.
  pub at: Option<Timestamp>,
}

/// Resolves a mention to a canonical entity, creating one when no candidate
/// passes the threshold.
///
/// `now` stamps a newly created entity; it is the only use of the clock (no
/// time is fabricated for scoring).
pub async fn resolve(
  store: &dyn Store,
  bank: BankId,
  mention: &EntityMention,
  ctx: &MentionContext,
  weights: &ResolveWeights,
  now: Timestamp,
) -> Result<EntityId> {
  let canonical = normalize_name(&mention.text);
  if canonical.is_empty() {
    return Err(Error::config("entity mention is empty after normalization"));
  }

  let candidates = store
    .entity_candidates(bank, &canonical, weights.candidate_limit)
    .await?;

  if let Some(exact) = candidates
    .iter()
    .find(|candidate| candidate.entity.canonical_name == canonical)
  {
    return Ok(exact.entity.id);
  }

  let mut best: Option<(f32, EntityId)> = None;
  for candidate in &candidates {
    let score = score_candidate(&canonical, candidate, ctx, weights);
    if best.is_none_or(|(best_score, _)| score > best_score) {
      best = Some((score, candidate.entity.id));
    }
  }

  if let Some((score, id)) = best
    && score >= weights.threshold
  {
    return Ok(id);
  }

  let new = NewEntity {
    id: EntityId::new(),
    bank_id: bank,
    canonical_name: canonical,
    entity_type: mention.kind.unwrap_or(EntityKind::Other),
    created_at: now,
  };
  let (id, _created) = store.ensure_entity(new).await?;

  Ok(id)
}

/// Lowercases, trims, and collapses whitespace.
pub fn normalize_name(value: &str) -> String {
  value
    .split_whitespace()
    .collect::<Vec<_>>()
    .join(" ")
    .to_lowercase()
}

/// Normalized Levenshtein similarity in `[0, 1]` (1 = identical).
pub fn string_sim(a: &str, b: &str) -> f32 {
  if a == b {
    return 1.0;
  }

  let a: Vec<char> = a.chars().collect();
  let b: Vec<char> = b.chars().collect();
  if a.is_empty() || b.is_empty() {
    return 0.0;
  }

  let distance = levenshtein(&a, &b);
  let longest = a.len().max(b.len());
  1.0 - (distance as f32 / longest as f32)
}

/// Overlap coefficient between the context entities and the candidate's
/// co-mentioned entities. Returns 0 when the context carries no entities.
pub fn cooccurrence_sim(candidate: &EntityCandidate, ctx: &MentionContext) -> f32 {
  if ctx.co_mentioned.is_empty() {
    return 0.0;
  }

  let hits = ctx
    .co_mentioned
    .iter()
    .filter(|entity| candidate.co_mentioned.contains(entity))
    .count();

  hits as f32 / ctx.co_mentioned.len() as f32
}

/// Exponential temporal proximity; 0 when either timestamp is unknown.
pub fn temporal_sim(candidate: &EntityCandidate, at: Option<Timestamp>, scale: Duration) -> f32 {
  match (at, candidate.last_mention_at) {
    (Some(at), Some(last)) => {
      let delta = (at.timestamp_millis() - last.timestamp_millis()).unsigned_abs() as f64;
      let scale = scale.as_millis().max(1) as f64;
      (-delta / scale).exp() as f32
    }
    _ => 0.0,
  }
}

fn score_candidate(
  normalized: &str,
  candidate: &EntityCandidate,
  ctx: &MentionContext,
  weights: &ResolveWeights,
) -> f32 {
  let mut total = weights.alpha * best_string_sim(normalized, candidate);
  let mut total_weight = weights.alpha;

  if !ctx.co_mentioned.is_empty() {
    total += weights.beta * cooccurrence_sim(candidate, ctx);
    total_weight += weights.beta;
  }
  if ctx.at.is_some() && candidate.last_mention_at.is_some() {
    total += weights.gamma * temporal_sim(candidate, ctx.at, weights.temporal_scale);
    total_weight += weights.gamma;
  }

  // Renormalize over the features that were actually available, so a missing
  // co-occurrence/temporal signal does not systematically lower the score.
  if total_weight > 0.0 {
    total / total_weight
  } else {
    0.0
  }
}

fn best_string_sim(normalized: &str, candidate: &EntityCandidate) -> f32 {
  let mut best = string_sim(normalized, &candidate.entity.canonical_name);
  for variant in &candidate.variants {
    best = best.max(string_sim(normalized, &normalize_name(variant)));
  }
  best
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
  let mut previous: Vec<usize> = (0..=b.len()).collect();
  let mut current = vec![0_usize; b.len() + 1];

  for (i, left) in a.iter().enumerate() {
    current[0] = i + 1;
    for (j, right) in b.iter().enumerate() {
      let cost = usize::from(left != right);
      current[j + 1] = (previous[j + 1] + 1)
        .min(current[j] + 1)
        .min(previous[j] + cost);
    }
    std::mem::swap(&mut previous, &mut current);
  }

  previous[b.len()]
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::timestamp;
  use crate::types::Entity;

  fn candidate(
    name: &str,
    variants: &[&str],
    last_mention_at: Option<Timestamp>,
  ) -> EntityCandidate {
    EntityCandidate {
      entity: Entity {
        id: EntityId::new(),
        bank_id: BankId::new(),
        canonical_name: name.into(),
        entity_type: EntityKind::Person,
        mention_count: 1,
        first_seen: None,
        last_seen: last_mention_at,
        attrs: None,
        created_at: timestamp(0),
        updated_at: timestamp(0),
      },
      variants: variants.iter().map(|value| (*value).to_string()).collect(),
      co_mentioned: Vec::new(),
      last_mention_at,
    }
  }

  #[test]
  fn normalizes_case_whitespace_and_cjk() {
    assert_eq!(normalize_name("  Emily   SMITH "), "emily smith");
    assert_eq!(normalize_name("张三"), "张三");
    assert_eq!(normalize_name(" \n\t "), "");
  }

  #[test]
  fn string_similarity_is_normalized_levenshtein() {
    assert_eq!(string_sim("emily", "emily"), 1.0);
    assert_eq!(string_sim("", "emily"), 0.0);
    assert_eq!(string_sim("emily", ""), 0.0);
    let typo = string_sim("emily smith", "emily smyth");
    assert!((typo - 10.0 / 11.0).abs() < 1e-6, "score = {typo}");
    assert_eq!(string_sim("张三", "李四"), 0.0);
    assert!((string_sim("张三", "张四") - 0.5).abs() < 1e-6);
  }

  #[test]
  fn cooccurrence_uses_the_context_overlap() {
    let mut cand = candidate("emily", &[], None);
    let one = EntityId::new();
    let two = EntityId::new();
    cand.co_mentioned = vec![one];

    let mut ctx = MentionContext {
      co_mentioned: vec![one, two],
      at: None,
    };
    assert_eq!(cooccurrence_sim(&cand, &ctx), 0.5);

    ctx.co_mentioned = vec![one];
    assert_eq!(cooccurrence_sim(&cand, &ctx), 1.0);

    ctx.co_mentioned.clear();
    assert_eq!(cooccurrence_sim(&cand, &ctx), 0.0);
  }

  #[test]
  fn temporal_similarity_decays_exponentially() {
    let scale = Duration::from_secs(30 * 24 * 60 * 60);
    let at = timestamp(1_717_934_000_000);
    let cand = candidate("emily", &[], Some(at));
    assert_eq!(temporal_sim(&cand, Some(at), scale), 1.0);

    let later = timestamp(1_717_934_000_000 + 30 * 24 * 60 * 60 * 1000);
    let decay = temporal_sim(&cand, Some(later), scale);
    assert!((decay - (-1.0_f32).exp()).abs() < 1e-6, "decay = {decay}");

    assert_eq!(temporal_sim(&cand, None, scale), 0.0);
    assert_eq!(
      temporal_sim(&candidate("emily", &[], None), Some(at), scale),
      0.0
    );
  }

  #[test]
  fn score_renormalizes_over_available_features() {
    let weights = ResolveWeights::default();
    let cand = candidate("emily smith", &[], None);
    let ctx = MentionContext::default();

    let alone = score_candidate("emily smith", &cand, &ctx, &weights);
    assert!(
      (alone - 1.0).abs() < 1e-6,
      "only string similarity is available"
    );

    let typo = score_candidate("emily smyth", &cand, &ctx, &weights);
    assert!(
      (typo - 10.0 / 11.0).abs() < 1e-6,
      "missing features are renormalized out"
    );
  }

  #[test]
  fn score_uses_variants_for_string_similarity() {
    let weights = ResolveWeights::default();
    let cand = candidate("emily smith", &["Emily S."], None);
    let ctx = MentionContext::default();

    let score = score_candidate("emily s", &cand, &ctx, &weights);
    assert!(score > 0.8, "variant boosts the score: {score}");
  }
}
