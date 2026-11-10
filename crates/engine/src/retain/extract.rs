//! Extraction call: A.1 prompt + JSON Schema + per-fact validation.
//!
//! Model calls happen outside any transaction; this module only produces
//! proposals. Persistence is the caller's job (the chunk commit).

use std::collections::HashSet;

use schemars::schema_for;
use tracing::warn;

use super::augment::normalize_time;
use crate::ports::{JsonRequest, Llm, LlmExt};
use crate::types::{
  Chunk, ExtractedFact, FactExtractionResponse, FactType, IntervalError, OccurrenceInterval,
};
use crate::{Error, Result};

/// The A.1 system prompt, embedded at compile time so the binary never reads
/// files at runtime.
const EXTRACTION_PROMPT: &str = include_str!("prompts/extraction.md");

/// Extraction call parameters. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractionConfig {
  /// Sampling temperature; extraction is deterministic, so `0.0` is the default.
  pub temperature: f32,
  /// Maximum number of tokens to generate.
  pub max_tokens: u32,
  /// Safety net in characters: longer chunk text is truncated with a warning.
  /// `0` disables truncation (chunking should already bound the size).
  pub max_input_chars: usize,
  /// Fallback belief strength when the model omits or misstates the confidence
  /// of an opinion fact. Must be in `[0, 1]`; out-of-range values are replaced
  /// by `0.5`.
  pub default_opinion_confidence: f32,
}

impl Default for ExtractionConfig {
  fn default() -> Self {
    Self {
      temperature: 0.0,
      max_tokens: 4096,
      max_input_chars: 100_000,
      default_opinion_confidence: 0.5,
    }
  }
}

/// The usable proposals of one extraction call.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractionReport {
  /// Facts that passed validation, with causal indices remapped to this vector.
  pub facts: Vec<ExtractedFact>,
  /// Facts dropped because they failed validation.
  pub discarded: usize,
  /// Durable first-person background snippets, normalized (trimmed, non-empty,
  /// deduplicated). They are merge input, not stored facts (paper §5.6).
  pub background_snippets: Vec<String>,
}

/// Extracts facts and background snippets from one chunk.
///
/// Each fact proposal first receives a mention time from the chunk
/// (`normalize_time`; occurrence bounds are never fabricated), then is validated
/// fact by fact: invalid facts (empty text, observation rows, reversed or
/// end-only occurrence intervals) are counted and dropped, and causal references
/// are remapped to the surviving list (references to dropped facts, out-of-range
/// indices, and self-references are removed). Background snippets are trimmed,
/// deduplicated, and returned separately; they are merge input, not facts.
/// Transport errors (timeouts, malformed JSON, schema mismatches) propagate
/// unchanged, so the caller can retry only [`crate::Error::Retryable`].
pub async fn extract_facts(
  llm: &dyn Llm,
  chunk: &Chunk,
  cfg: &ExtractionConfig,
) -> Result<ExtractionReport> {
  let schema = serde_json::to_value(schema_for!(FactExtractionResponse))
    .map_err(|error| Error::fatal(format!("serialize extraction schema: {error}")))?;

  let request = JsonRequest {
    system: Some(EXTRACTION_PROMPT.to_string()),
    user: truncate_input(&chunk.text, cfg.max_input_chars),
    schema_name: "fact_extraction",
    schema,
    temperature: cfg.temperature,
    max_tokens: cfg.max_tokens,
  };

  let response: FactExtractionResponse = llm.complete_json(request).await?;

  let mut report = filter_facts(response.facts, chunk, cfg.default_opinion_confidence);
  report.background_snippets = normalize_background_snippets(response.background_snippets);
  Ok(report)
}

/// Returns a reason when a single extracted fact must be discarded.
///
/// The occurrence checks (no end-only interval, no reversed bounds) apply to the
/// bounds reported by the model; `mentioned_at` is not inspected.
/// [`normalize_time`] only fills the mention time, so there is no ordering
/// requirement between the two functions.
///
/// `fact_type` is already validated by deserialization; the rest are semantic
/// rules. Opinion facts require a finite `confidence` in `[0, 1]`; non-opinion
/// facts must not carry one. An open interval (`start` only) is valid: the event
/// is still ongoing.
pub fn validate_extracted(fact: &ExtractedFact) -> Option<&'static str> {
  if fact.what.trim().is_empty() {
    return Some("empty `what`");
  }
  if fact.fact_type == FactType::Observation {
    return Some("observation rows are synthesized, not extracted");
  }
  if fact.fact_type == FactType::Opinion && fact.confidence.is_none() {
    return Some("opinion facts require `confidence`");
  }
  if fact.fact_type != FactType::Opinion && fact.confidence.is_some() {
    return Some("`confidence` is only valid on opinion facts");
  }
  if let Some(confidence) = fact.confidence
    && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
  {
    return Some("`confidence` outside [0, 1]");
  }
  if let Err(error) = OccurrenceInterval::from_parts(fact.occurred_start, fact.occurred_end) {
    return Some(match error {
      IntervalError::Reversed { .. } => "`occurred_end` precedes `occurred_start`",
      IntervalError::EndWithoutStart { .. } => "`occurred_end` without `occurred_start`",
    });
  }
  None
}

fn normalize_confidence(fact: &mut ExtractedFact, default: f32) {
  if fact.fact_type != FactType::Opinion {
    if fact.confidence.is_some() {
      warn!(
        what = %fact.what,
        "dropping confidence from a non-opinion fact"
      );
      fact.confidence = None;
    }
    return;
  }

  let fallback = if default.is_finite() && (0.0..=1.0).contains(&default) {
    default
  } else {
    0.5
  };

  match fact.confidence {
    Some(value) if value.is_finite() && (0.0..=1.0).contains(&value) => {}
    Some(value) => {
      warn!(
        what = %fact.what,
        value = %value,
        fallback,
        "invalid opinion confidence; using the fallback"
      );
      fact.confidence = Some(fallback);
    }
    None => {
      warn!(
        what = %fact.what,
        fallback,
        "missing opinion confidence; using the fallback"
      );
      fact.confidence = Some(fallback);
    }
  }
}

fn filter_facts(
  facts: Vec<ExtractedFact>,
  chunk: &Chunk,
  default_opinion_confidence: f32,
) -> ExtractionReport {
  let original_len = facts.len();
  let mut old_to_new: Vec<Option<usize>> = vec![None; original_len];
  let mut kept: Vec<(usize, ExtractedFact)> = Vec::with_capacity(original_len);
  let mut discarded = 0_usize;

  for (original_index, mut fact) in facts.into_iter().enumerate() {
    // Normalize the mention time first; occurrence bounds are never fabricated.
    normalize_time(&mut fact, chunk);
    normalize_confidence(&mut fact, default_opinion_confidence);

    if let Some(reason) = validate_extracted(&fact) {
      warn!(
        index = original_index,
        reason, "discarding invalid extracted fact"
      );
      discarded += 1;
      continue;
    }

    // Empty mention text carries no resolution signal.
    fact
      .entities
      .retain(|entity| !entity.text.trim().is_empty());

    old_to_new[original_index] = Some(kept.len());
    kept.push((original_index, fact));
  }

  // Causal indices refer to the original `facts` array; rewrite them to the
  // surviving list, dropping references that cannot be resolved.
  for (new_index, (original_index, fact)) in kept.iter_mut().enumerate() {
    fact.causal_relations.retain_mut(|relation| {
      let target = relation.target_fact_index;
      match old_to_new.get(target).copied().flatten() {
        Some(mapped) if mapped != new_index => {
          relation.target_fact_index = mapped;
          true
        }
        _ => {
          warn!(
            original_index,
            target, "dropping an unresolvable causal reference"
          );
          false
        }
      }
    });
  }

  ExtractionReport {
    facts: kept.into_iter().map(|(_, fact)| fact).collect(),
    discarded,
    background_snippets: Vec::new(),
  }
}

/// Normalizes model-proposed background snippets: trim, drop empty entries,
/// deduplicate, and preserve the model's order.
fn normalize_background_snippets(snippets: Vec<String>) -> Vec<String> {
  let mut seen = HashSet::new();
  snippets
    .into_iter()
    .map(|snippet| snippet.trim().to_string())
    .filter(|snippet| !snippet.is_empty())
    .filter(|snippet| seen.insert(snippet.clone()))
    .collect()
}

fn truncate_input(text: &str, max_chars: usize) -> String {
  if max_chars == 0 {
    return text.to_string();
  }

  let chars = text.chars().count();
  if chars <= max_chars {
    return text.to_string();
  }

  warn!(
    chars,
    max_chars, "extraction input exceeded the character limit; truncating"
  );
  text.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
  use async_trait::async_trait;
  use serde_json::{Value, json};

  use super::*;
  use crate::ports::JsonRequest;
  use crate::testing::{FakeLlm, timestamp};
  use crate::types::{BankId, ChunkId, ChunkStatus, SourceId};

  fn chunk(text: &str) -> Chunk {
    Chunk {
      id: ChunkId::new(),
      bank_id: BankId::new(),
      source_id: SourceId::new(),
      chunk_index: 0,
      text: text.into(),
      content_hash: "hash".into(),
      status: ChunkStatus::Pending,
      time_start: None,
      time_end: None,
      created_at: timestamp(0),
      updated_at: timestamp(0),
    }
  }

  fn chunk_with_time(text: &str, start: Option<i64>, end: Option<i64>) -> Chunk {
    Chunk {
      time_start: start.map(timestamp),
      time_end: end.map(timestamp),
      ..chunk(text)
    }
  }

  fn at(millis: i64) -> String {
    timestamp(millis).to_rfc3339()
  }

  fn fact_json(what: &str) -> Value {
    json!({
      "what": what,
      "when": "Saturday, June 9, 2024",
      "where": "Berlin",
      "who": "Emily",
      "why": "it matters",
      "fact_type": "world"
    })
  }

  fn fact_json_at(what: &str, start: Option<i64>, end: Option<i64>) -> Value {
    let mut value = fact_json(what);
    if let Some(start) = start {
      value["occurred_start"] = json!(at(start));
    }
    if let Some(end) = end {
      value["occurred_end"] = json!(at(end));
    }
    value
  }

  #[tokio::test]
  async fn discards_invalid_facts_and_records_the_request() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({ "facts": [fact_json("first"), fact_json("   "), fact_json("third")] }),
    );

    let report = extract_facts(&llm, &chunk("chunk text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert_eq!(report.facts.len(), 2);
    assert_eq!(report.discarded, 1);
    assert_eq!(report.facts[0].what, "first");
    assert_eq!(report.facts[1].what, "third");

    let calls = llm.calls();
    assert_eq!(calls.len(), 1);
    let request = &calls[0];
    assert_eq!(request.schema_name, "fact_extraction");
    assert!(request.user.contains("chunk text"));
    assert!(
      request
        .system
        .as_deref()
        .is_some_and(|prompt| prompt.contains("MAXIMUM VERBOSITY"))
    );
    assert!(request.schema.get("properties").is_some());
    assert_eq!(request.temperature, 0.0);
  }

  #[tokio::test]
  async fn remaps_causal_indices_to_the_surviving_facts() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({
        "facts": [
          {
            "what": "first", "when": "w", "where": "w", "who": "w", "why": "w",
            "fact_type": "world",
            "causal_relations": [
              { "target_fact_index": 2, "relation": "causes", "strength": 0.5 },
              { "target_fact_index": 1, "relation": "enables", "strength": 0.5 }
            ]
          },
          { "what": "", "when": "w", "where": "w", "who": "w", "why": "w", "fact_type": "world" },
          {
            "what": "third", "when": "w", "where": "w", "who": "w", "why": "w",
            "fact_type": "world",
            "causal_relations": [
              { "target_fact_index": 0, "relation": "caused_by", "strength": 0.5 }
            ]
          }
        ]
      }),
    );

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert_eq!(report.discarded, 1);
    assert_eq!(report.facts.len(), 2);
    // old 0 -> new 0; old 2 -> new 1.
    assert_eq!(report.facts[0].causal_relations.len(), 1);
    assert_eq!(report.facts[0].causal_relations[0].target_fact_index, 1);
    assert_eq!(report.facts[1].causal_relations[0].target_fact_index, 0);
  }

  #[tokio::test]
  async fn self_references_are_dropped() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({
        "facts": [{
          "what": "only", "when": "w", "where": "w", "who": "w", "why": "w",
          "fact_type": "world",
          "causal_relations": [
            { "target_fact_index": 0, "relation": "causes", "strength": 1.0 }
          ]
        }]
      }),
    );

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert_eq!(report.facts.len(), 1);
    assert!(report.facts[0].causal_relations.is_empty());
  }

  #[tokio::test]
  async fn rejects_observation_facts_and_empty_entities() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({
        "facts": [
          {
            "what": "world", "when": "w", "where": "w", "who": "w", "why": "w",
            "fact_type": "world",
            "entities": [{ "text": "  ", "kind": "person" }, { "text": "Emily", "kind": "person" }]
          },
          { "what": "obs", "when": "w", "where": "w", "who": "w", "why": "w", "fact_type": "observation" }
        ]
      }),
    );

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert_eq!(report.facts.len(), 1);
    assert_eq!(report.discarded, 1);
    assert_eq!(report.facts[0].entities.len(), 1);
    assert_eq!(report.facts[0].entities[0].text, "Emily");
  }

  fn opinion_json(what: &str, confidence: Option<f64>) -> Value {
    let mut value = fact_json(what);
    value["fact_type"] = json!("opinion");
    if let Some(confidence) = confidence {
      value["confidence"] = json!(confidence);
    }
    value
  }

  #[tokio::test]
  async fn keeps_an_opinion_confidence_from_the_model() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({ "facts": [opinion_json("I like Rust", Some(0.8))] }),
    );

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert_eq!(report.facts.len(), 1);
    assert_eq!(report.facts[0].fact_type, FactType::Opinion);
    assert_eq!(report.facts[0].confidence, Some(0.8));
  }

  #[tokio::test]
  async fn defaults_a_missing_opinion_confidence() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({ "facts": [opinion_json("I like Rust", None)] }),
    );
    let cfg = ExtractionConfig {
      default_opinion_confidence: 0.4,
      ..ExtractionConfig::default()
    };

    let report = extract_facts(&llm, &chunk("text"), &cfg)
      .await
      .expect("extraction");

    assert_eq!(report.facts[0].confidence, Some(0.4));
  }

  #[tokio::test]
  async fn falls_back_for_an_invalid_opinion_confidence() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({ "facts": [opinion_json("I like Rust", Some(1.5))] }),
    );

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert_eq!(report.facts[0].confidence, Some(0.5));
  }

  #[tokio::test]
  async fn drops_confidence_from_non_opinion_facts() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({ "facts": [{
        "what": "world fact", "when": "w", "where": "w", "who": "w", "why": "w",
        "fact_type": "world", "confidence": 0.9
      }] }),
    );

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert_eq!(report.facts.len(), 1);
    assert_eq!(report.facts[0].confidence, None);
  }

  #[tokio::test]
  async fn empty_result_is_ok() {
    let llm = FakeLlm::new();
    llm.push("fact_extraction", json!({ "facts": [] }));

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert!(report.facts.is_empty());
    assert_eq!(report.discarded, 0);
  }

  #[tokio::test]
  async fn truncates_oversized_input_at_char_boundaries() {
    let llm = FakeLlm::new();
    llm.push("fact_extraction", json!({ "facts": [] }));
    let cfg = ExtractionConfig {
      max_input_chars: 5,
      ..ExtractionConfig::default()
    };

    extract_facts(&llm, &chunk("你好世界你好"), &cfg)
      .await
      .expect("extraction");

    assert_eq!(llm.calls()[0].user, "你好世界你");
  }

  #[tokio::test]
  async fn schema_mismatch_is_fatal() {
    let llm = FakeLlm::new();
    llm.push("fact_extraction", json!({}));

    let error = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect_err("missing facts");

    assert!(matches!(error, Error::Fatal(_)));
    assert!(!error.is_retryable());
  }

  struct UnavailableLlm;

  #[async_trait]
  impl Llm for UnavailableLlm {
    async fn complete_json_raw(&self, _req: JsonRequest) -> Result<Value> {
      Err(Error::retryable("provider timeout"))
    }

    fn model(&self) -> &str {
      "unavailable"
    }
  }

  #[tokio::test]
  async fn retryable_errors_propagate_unchanged() {
    let error = extract_facts(
      &UnavailableLlm,
      &chunk("text"),
      &ExtractionConfig::default(),
    )
    .await
    .expect_err("timeout");

    assert!(error.is_retryable());
  }

  #[tokio::test]
  async fn keeps_future_open_intervals_and_discards_end_only() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({
        "facts": [
          // A future start is a valid open interval; no end is invented.
          fact_json_at("future", Some(25_000), None),
          // End-only has no representable interval and is discarded.
          fact_json_at("end only", None, Some(15_000)),
          // No times: occurrence stays unknown; only the mention time is filled.
          fact_json("untimed")
        ]
      }),
    );

    let report = extract_facts(
      &llm,
      &chunk_with_time("text", Some(10_000), Some(20_000)),
      &ExtractionConfig::default(),
    )
    .await
    .expect("extraction");

    assert_eq!(report.discarded, 1);
    assert_eq!(report.facts.len(), 2);

    assert_eq!(report.facts[0].what, "future");
    assert_eq!(report.facts[0].occurred_start, Some(timestamp(25_000)));
    assert_eq!(report.facts[0].occurred_end, None);
    assert_eq!(report.facts[0].mentioned_at, Some(timestamp(15_000)));

    assert_eq!(report.facts[1].what, "untimed");
    assert_eq!(report.facts[1].occurred_start, None);
    assert_eq!(report.facts[1].occurred_end, None);
    assert_eq!(report.facts[1].mentioned_at, Some(timestamp(15_000)));
  }

  #[tokio::test]
  async fn discards_end_only_intervals_without_chunk_time() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({
        "facts": [
          fact_json_at("end only", None, Some(15_000)),
          fact_json_at("open", Some(5_000), None)
        ]
      }),
    );

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert_eq!(report.discarded, 1);
    assert_eq!(report.facts.len(), 1);
    assert_eq!(report.facts[0].what, "open");
    assert_eq!(report.facts[0].occurred_start, Some(timestamp(5_000)));
    assert_eq!(
      report.facts[0].occurred_end, None,
      "an open interval is valid"
    );
  }

  #[tokio::test]
  async fn discarding_a_reversed_fact_drops_causal_references_to_it() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({
        "facts": [
          fact_json_at("reversed", Some(25_000), Some(20_000)),
          {
            "what": "references the bad fact", "when": "w", "where": "w", "who": "w",
            "why": "w", "fact_type": "world",
            "causal_relations": [
              { "target_fact_index": 0, "relation": "causes", "strength": 1.0 }
            ]
          }
        ]
      }),
    );

    let report = extract_facts(
      &llm,
      &chunk_with_time("text", Some(10_000), Some(20_000)),
      &ExtractionConfig::default(),
    )
    .await
    .expect("extraction");

    assert_eq!(report.discarded, 1);
    assert_eq!(report.facts.len(), 1);
    assert!(
      report.facts[0].causal_relations.is_empty(),
      "the reference to the discarded fact is dropped"
    );
  }

  #[tokio::test]
  async fn normalizes_background_snippets() {
    let llm = FakeLlm::new();
    llm.push(
      "fact_extraction",
      json!({
        "facts": [],
        "background_snippets": [
          "  I was born in Colorado.  ",
          "",
          "I was born in Colorado.",
          "I have ten years of startup experience."
        ]
      }),
    );

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert_eq!(
      report.background_snippets,
      vec![
        "I was born in Colorado.".to_string(),
        "I have ten years of startup experience.".to_string()
      ]
    );
  }

  #[tokio::test]
  async fn background_snippets_default_to_empty() {
    let llm = FakeLlm::new();
    llm.push("fact_extraction", json!({ "facts": [] }));

    let report = extract_facts(&llm, &chunk("text"), &ExtractionConfig::default())
      .await
      .expect("extraction");

    assert!(report.background_snippets.is_empty());
  }
}
