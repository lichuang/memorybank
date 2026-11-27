//! Opinion persistence: binding known entities and writing the row (paper §5.4).
//!
//! Reflect produces free first-person text; the entity set `E` of an opinion is
//! recovered by matching known canonical names and their observed variants. This
//! is deliberately not an LLM call: resolution already ran during retain, and an
//! opinion should attach to existing entities rather than invent new ones.

use crate::ports::{NewOpinion, Store};
use crate::types::{BankId, EntityId, FactId, Timestamp};
use crate::{Error, Result};

/// Upper bound on entity candidates scanned when binding opinion text.
const CANDIDATE_LIMIT: usize = 500;

/// Matches known entities inside `text`.
///
/// Matching is a case-insensitive substring search over each candidate's
/// canonical name and recorded `mention_text` variants, longest surface form
/// first, so "Emily Smith" wins over "Emily" on the same span. Overlapping matches
/// resolve to the longest form, and each entity is bound once no matter how often
/// it is named. No entity is created: an opinion never introduces an entity that
/// retain has not seen.
pub async fn bind_known_entities(
  store: &dyn Store,
  bank: BankId,
  text: &str,
) -> Result<Vec<(EntityId, Option<String>)>> {
  if text.trim().is_empty() {
    return Ok(Vec::new());
  }

  // Seeding with the whole lowered text returns exact/variant hits first and then
  // the bank's most-mentioned entities — the same bounded shortlist that entity
  // resolution uses.
  let candidates = store
    .entity_candidates(bank, &text.to_lowercase(), CANDIDATE_LIMIT)
    .await?;
  if candidates.is_empty() {
    return Ok(Vec::new());
  }

  let haystack = text.to_lowercase();
  let mut surface_forms: Vec<(EntityId, String)> = Vec::new();
  for candidate in &candidates {
    let canonical = candidate.entity.canonical_name.trim().to_lowercase();
    if canonical.is_empty() {
      continue;
    }
    surface_forms.push((candidate.entity.id, canonical.clone()));
    for variant in &candidate.variants {
      let variant = variant.trim().to_lowercase();
      if variant.is_empty() || variant == canonical {
        continue;
      }
      surface_forms.push((candidate.entity.id, variant));
    }
  }
  // Longest surface first so a short name cannot shadow a longer one.
  surface_forms.sort_by(|left, right| right.1.chars().count().cmp(&left.1.chars().count()));

  let mut claimed: Vec<(usize, usize)> = Vec::new();
  let mut bindings: Vec<(EntityId, Option<String>)> = Vec::new();

  for (entity_id, needle) in &surface_forms {
    if bindings.iter().any(|(bound, _)| bound == entity_id) {
      continue;
    }

    let mut from = 0;
    while let Some(offset) = haystack[from..].find(needle.as_str()) {
      let start = from + offset;
      let end = start + needle.len();
      from = end.max(from + 1);

      if claimed
        .iter()
        .any(|(claimed_start, claimed_end)| start < *claimed_end && end > *claimed_start)
      {
        continue;
      }
      claimed.push((start, end));
      bindings.push((*entity_id, Some(needle.clone())));
      break;
    }
  }

  Ok(bindings)
}

/// Builds the opinion row for persistence.
///
/// The text is stored verbatim — opinions are not events, so no time prefix is
/// added. The id is pre-allocated so a retry cannot duplicate the row, and
/// `chunk_id` stays `None` (the store inserts reflect-formed opinions without a
/// chunk). Confidence is clamped defensively; callers should already have
/// normalized it.
#[allow(clippy::too_many_arguments)]
pub fn build_opinion(
  id: FactId,
  bank: BankId,
  text: &str,
  confidence: f32,
  entities: Vec<(EntityId, Option<String>)>,
  embedding: Option<Vec<f32>>,
  created_at: Timestamp,
) -> Result<NewOpinion> {
  if text.trim().is_empty() {
    return Err(Error::config("opinion text must not be empty"));
  }
  if !confidence.is_finite() {
    return Err(Error::config("opinion confidence must be finite"));
  }

  Ok(NewOpinion {
    id,
    bank_id: bank,
    text: text.to_string(),
    confidence: confidence.clamp(0.0, 1.0),
    entities,
    embedding,
    created_at,
  })
}

/// Binds known entities and returns the row, ready for [`Store::insert_opinion`].
#[allow(clippy::too_many_arguments)]
pub async fn prepare_opinion(
  store: &dyn Store,
  id: FactId,
  bank: BankId,
  text: &str,
  confidence: f32,
  embedding: Option<Vec<f32>>,
  created_at: Timestamp,
) -> Result<NewOpinion> {
  let entities = bind_known_entities(store, bank, text).await?;
  build_opinion(id, bank, text, confidence, entities, embedding, created_at)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::timestamp;

  #[test]
  fn build_opinion_rejects_empty_text_and_non_finite_confidence() {
    let bank = BankId::new();
    let id = FactId::new();

    assert!(matches!(
      build_opinion(id, bank, "   ", 0.5, Vec::new(), None, timestamp(0)),
      Err(Error::Config(_))
    ));
    assert!(matches!(
      build_opinion(
        id,
        bank,
        "I think A",
        f32::NAN,
        Vec::new(),
        None,
        timestamp(0)
      ),
      Err(Error::Config(_))
    ));
  }

  #[test]
  fn build_opinion_clamps_confidence_and_keeps_text_verbatim() {
    let opinion = build_opinion(
      FactId::new(),
      BankId::new(),
      "I think A",
      1.7,
      Vec::new(),
      None,
      timestamp(0),
    )
    .expect("opinion");

    assert_eq!(opinion.confidence, 1.0);
    assert_eq!(opinion.text, "I think A");
    assert!(opinion.embedding.is_none());
  }

  #[test]
  fn build_opinion_accepts_an_embedding_and_bindings() {
    let entity = EntityId::new();
    let opinion = build_opinion(
      FactId::new(),
      BankId::new(),
      "I believe Alice is reliable",
      0.4,
      vec![(entity, Some("alice".into()))],
      Some(vec![0.1, 0.2]),
      timestamp(5),
    )
    .expect("opinion");

    assert_eq!(opinion.confidence, 0.4);
    assert_eq!(opinion.entities, vec![(entity, Some("alice".into()))]);
    assert_eq!(opinion.embedding, Some(vec![0.1, 0.2]));
    assert_eq!(opinion.created_at, timestamp(5));
  }
}
