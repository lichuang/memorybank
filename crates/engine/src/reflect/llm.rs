//! One reflect LLM call: answer plus candidate opinions (paper §5.4.2, A.2).
//!
//! The call is a single structured request: the caller has already assembled the
//! system message (profile + numbered evidence), so this module only sends it,
//! validates the response, and normalizes the opinion drafts. Nothing is
//! persisted here — opinion rows are written later, outside this call.

use schemars::schema_for;

use crate::ports::{JsonRequest, Llm, LlmExt};
use crate::types::ReflectResponse;
use crate::{Error, Result};

/// The reflect user prompt, embedded at compile time.
const REFLECT_PROMPT: &str = include_str!("prompts/reflect.md");

/// Reflect call parameters. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflectConfig {
  /// Sampling temperature. Reflection is opinionated by design, so this is
  /// higher than extraction's `0.0`.
  pub temperature: f32,
  /// Maximum number of tokens to generate.
  pub max_tokens: u32,
  /// Fallback confidence when the model returns a non-finite or out-of-range
  /// value.
  pub default_opinion_confidence: f32,
}

impl Default for ReflectConfig {
  fn default() -> Self {
    Self {
      temperature: 0.3,
      max_tokens: 1_500,
      default_opinion_confidence: 0.5,
    }
  }
}

/// Runs one reflect call.
///
/// `system` is the assembled system message and `query` is the user's question.
/// A response is accepted when it deserializes as [`ReflectResponse`]; an empty
/// `answer` is rejected as [`Error::Fatal`] because it means the model violated
/// the contract, while an empty `opinions` list is valid and common.
///
/// Opinion drafts are normalized in place: confidence is clamped to `[0, 1]`
/// (non-finite values fall back to `cfg.default_opinion_confidence`, itself
/// clamped), and drafts with an empty opinion text are dropped with a warning.
/// Missing entity bindings are not inferred here; persistence resolves them.
pub async fn reflect_llm(
  llm: &dyn Llm,
  system: String,
  query: &str,
  cfg: &ReflectConfig,
) -> Result<ReflectResponse> {
  let schema = serde_json::to_value(schema_for!(ReflectResponse))
    .map_err(|error| Error::fatal(format!("serialize reflect schema: {error}")))?;

  let request = JsonRequest {
    system: Some(system),
    user: format!("QUESTION:\n{query}\n\n{REFLECT_PROMPT}"),
    schema_name: "reflect",
    schema,
    temperature: cfg.temperature,
    max_tokens: cfg.max_tokens,
  };

  let mut response: ReflectResponse = llm.complete_json(request).await?;

  if response.answer.trim().is_empty() {
    return Err(Error::fatal("reflect returned an empty answer"));
  }

  let fallback = normalize_fallback(cfg.default_opinion_confidence);
  response.opinions.retain_mut(|draft| {
    if draft.opinion.trim().is_empty() {
      tracing::warn!("dropping a reflect opinion with empty text");
      return false;
    }
    draft.confidence = normalize_confidence(draft.confidence, fallback);
    true
  });

  Ok(response)
}

/// Returns a usable fallback confidence.
fn normalize_fallback(value: f32) -> f32 {
  if value.is_finite() && (0.0..=1.0).contains(&value) {
    value
  } else {
    0.5
  }
}

/// Clamps one opinion's confidence.
fn normalize_confidence(value: f32, fallback: f32) -> f32 {
  if value.is_finite() {
    value.clamp(0.0, 1.0)
  } else {
    fallback
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::FakeLlm;
  use crate::types::OpinionDraft;
  use serde_json::json;

  fn draft(opinion: &str, confidence: f64) -> serde_json::Value {
    json!({
      "opinion": opinion,
      "confidence": confidence,
      "reasoning": "because the evidence supports it"
    })
  }

  #[tokio::test]
  async fn returns_the_answer_and_two_opinions() {
    let llm = FakeLlm::new();
    llm.push(
      "reflect",
      json!({
        "answer": "Alice is reliable [1].",
        "opinions": [
          draft("I think Alice is reliable", 0.8),
          draft("I believe track records matter", 0.6)
        ]
      }),
    );

    let response = reflect_llm(
      &llm,
      "system message".into(),
      "Is Alice reliable?",
      &ReflectConfig::default(),
    )
    .await
    .expect("reflect");

    assert_eq!(response.answer, "Alice is reliable [1].");
    assert_eq!(response.opinions.len(), 2);
    assert_eq!(response.opinions[0].confidence, 0.8);

    let calls = llm.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].schema_name, "reflect");
    assert_eq!(calls[0].system.as_deref(), Some("system message"));
    assert!(calls[0].user.contains("Is Alice reliable?"));
  }

  #[tokio::test]
  async fn accepts_an_empty_opinion_list() {
    let llm = FakeLlm::new();
    llm.push(
      "reflect",
      json!({ "answer": "I do not have enough information.", "opinions": [] }),
    );

    let response = reflect_llm(&llm, "sys".into(), "q", &ReflectConfig::default())
      .await
      .expect("reflect");

    assert!(response.opinions.is_empty());
    assert!(!response.answer.is_empty());
  }

  #[tokio::test]
  async fn clamps_confidence_and_falls_back_on_non_finite_values() {
    let llm = FakeLlm::new();
    llm.push(
      "reflect",
      json!({
        "answer": "an answer",
        "opinions": [
          draft("I think A", 1.8),
          draft("I think B", -0.4),
          draft("I think C", 0.42)
        ]
      }),
    );

    let response = reflect_llm(&llm, "sys".into(), "q", &ReflectConfig::default())
      .await
      .expect("reflect");

    assert_eq!(response.opinions[0].confidence, 1.0);
    assert_eq!(response.opinions[1].confidence, 0.0);
    assert_eq!(response.opinions[2].confidence, 0.42);
  }

  #[tokio::test]
  async fn drops_empty_opinion_text_and_keeps_the_rest() {
    let llm = FakeLlm::new();
    llm.push(
      "reflect",
      json!({
        "answer": "an answer",
        "opinions": [
          draft("   ", 0.5),
          draft("I think A", 0.5)
        ]
      }),
    );

    let response = reflect_llm(&llm, "sys".into(), "q", &ReflectConfig::default())
      .await
      .expect("reflect");

    assert_eq!(response.opinions.len(), 1);
    assert_eq!(response.opinions[0].opinion, "I think A");
  }

  #[tokio::test]
  async fn rejects_an_empty_answer() {
    let llm = FakeLlm::new();
    llm.push("reflect", json!({ "answer": "   ", "opinions": [] }));

    let error = reflect_llm(&llm, "sys".into(), "q", &ReflectConfig::default())
      .await
      .expect_err("empty answer");

    assert!(matches!(error, Error::Fatal(_)), "error: {error}");
  }

  #[tokio::test]
  async fn propagates_retryable_transport_errors() {
    let llm = FakeLlm::new();

    let error = reflect_llm(&llm, "sys".into(), "q", &ReflectConfig::default())
      .await
      .expect_err("exhausted script");

    assert!(matches!(error, Error::Fatal(_)), "error: {error}");
  }

  #[tokio::test]
  async fn a_missing_opinions_key_is_a_contract_violation() {
    let llm = FakeLlm::new();
    llm.push("reflect", json!({ "answer": "no opinions key" }));

    let error = reflect_llm(&llm, "sys".into(), "q", &ReflectConfig::default())
      .await
      .expect_err("missing opinions");

    assert!(matches!(error, Error::Fatal(_)), "error: {error}");
  }

  #[test]
  fn invalid_fallback_values_revert_to_half() {
    for value in [f32::NAN, -1.0, 2.0, f32::INFINITY] {
      assert_eq!(normalize_fallback(value), 0.5);
    }
    assert_eq!(normalize_fallback(0.7), 0.7);
  }

  #[test]
  fn opinion_draft_shape_is_unchanged() {
    let draft = OpinionDraft {
      opinion: "I think A".into(),
      confidence: 0.5,
      reasoning: "reason".into(),
    };
    assert_eq!(draft.confidence, 0.5);
  }
}
