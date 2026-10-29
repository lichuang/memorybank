//! Extraction call: A.1 prompt + JSON Schema + per-fact validation.
//!
//! Model calls happen outside any transaction; this module only produces
//! proposals. Persistence is the caller's job (P1-16).

use schemars::schema_for;
use tracing::warn;

use crate::ports::{JsonRequest, Llm, LlmExt};
use crate::types::{Chunk, ExtractedFact, FactExtractionResponse, FactType};
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
}

impl Default for ExtractionConfig {
  fn default() -> Self {
    Self {
      temperature: 0.0,
      max_tokens: 4096,
      max_input_chars: 100_000,
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
}

/// Extracts facts from one chunk.
///
/// The response is validated fact by fact: invalid facts are counted and
/// dropped, and causal references are remapped to the surviving list (references
/// to dropped facts, out-of-range indices, and self-references are removed).
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

  Ok(filter_facts(response.facts))
}

/// Returns a reason when a single extracted fact must be discarded.
///
/// `fact_type` is already validated by deserialization; this checks semantic
/// rules only. Observation rows are never extraction products (paper A.5.1).
pub fn validate_extracted(fact: &ExtractedFact) -> Option<&'static str> {
  if fact.what.trim().is_empty() {
    return Some("empty `what`");
  }
  if fact.fact_type == FactType::Observation {
    return Some("observation rows are synthesized, not extracted");
  }
  None
}

fn filter_facts(facts: Vec<ExtractedFact>) -> ExtractionReport {
  let original_len = facts.len();
  let mut old_to_new: Vec<Option<usize>> = vec![None; original_len];
  let mut kept: Vec<(usize, ExtractedFact)> = Vec::with_capacity(original_len);
  let mut discarded = 0_usize;

  for (original_index, mut fact) in facts.into_iter().enumerate() {
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
  }
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
}
