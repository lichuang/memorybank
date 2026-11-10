//! Background merging (paper §5.6, retain step 5): `h' = MergeLLM(h, h_new)`.
//!
//! The model call stays outside any transaction. Its result is validated here
//! and later carried by [`crate::ports::ChunkCommit::background_update`], so the
//! store can apply it atomically with the chunk that produced the snippets.

use schemars::schema_for;

use crate::ports::{JsonRequest, Llm, LlmExt};
use crate::types::BackgroundMergeResponse;
use crate::{Error, Result};

/// The background-merge system prompt, embedded at compile time.
const BACKGROUND_MERGE_PROMPT: &str = include_str!("prompts/background_merge.md");

/// Background merge parameters. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct BackgroundConfig {
  /// Sampling temperature. Merging should be conservative, so `0.0` is the
  /// default.
  pub temperature: f32,
  /// Maximum number of tokens to generate.
  pub max_tokens: u32,
  /// Safety cap in characters for the merged background. `0` disables the cap.
  /// An over-long result is a contract violation, not a silent truncation.
  pub max_chars: usize,
  /// Total merge attempts (including the first) when the worker loses the
  /// background CAS to a concurrent chunk. Consumed by the worker; the merge
  /// function itself always performs one call.
  pub max_attempts: u8,
}

impl Default for BackgroundConfig {
  fn default() -> Self {
    Self {
      temperature: 0.0,
      max_tokens: 512,
      max_chars: 2_000,
      max_attempts: 3,
    }
  }
}

/// Merges `snippets` into `current` and returns the validated new background.
///
/// An empty snippet list is a no-op and does not call the model. The returned
/// string is trimmed and must be non-empty and within `cfg.max_chars`; an empty
/// or over-long result is [`Error::Fatal`] because it violates the response
/// contract rather than a transient transport condition.
pub async fn merge_background(
  llm: &dyn Llm,
  current: &str,
  snippets: &[String],
  cfg: &BackgroundConfig,
) -> Result<String> {
  if snippets.is_empty() {
    return Ok(current.to_string());
  }

  let schema = serde_json::to_value(schema_for!(BackgroundMergeResponse))
    .map_err(|error| Error::fatal(format!("serialize background merge schema: {error}")))?;

  let request = JsonRequest {
    system: Some(BACKGROUND_MERGE_PROMPT.to_string()),
    user: render_user(current, snippets),
    schema_name: "background_merge",
    schema,
    temperature: cfg.temperature,
    max_tokens: cfg.max_tokens,
  };

  let response: BackgroundMergeResponse = llm.complete_json(request).await?;
  let merged = response.background.trim().to_string();

  if merged.is_empty() {
    return Err(Error::fatal(
      "background merge returned an empty background",
    ));
  }
  if cfg.max_chars > 0 && merged.chars().count() > cfg.max_chars {
    return Err(Error::fatal(format!(
      "background merge exceeded {} characters (got {})",
      cfg.max_chars,
      merged.chars().count()
    )));
  }

  Ok(merged)
}

fn render_user(current: &str, snippets: &[String]) -> String {
  let mut user = String::from("CURRENT BACKGROUND:\n");
  let current = current.trim();
  if current.is_empty() {
    user.push_str("(empty)\n");
  } else {
    user.push_str(current);
    user.push('\n');
  }

  user.push_str("\nNEW BACKGROUND SNIPPETS:\n");
  for (index, snippet) in snippets.iter().enumerate() {
    user.push_str(&format!("{}. {}\n", index + 1, snippet));
  }
  user
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::FakeLlm;
  use serde_json::json;

  #[tokio::test]
  async fn merges_current_background_with_snippets() {
    let llm = FakeLlm::new();
    llm.push(
      "background_merge",
      json!({ "background": "I was born in Texas and have ten years of startup experience." }),
    );
    let snippets = vec![
      "I was born in Texas.".to_string(),
      "I have ten years of startup experience.".to_string(),
    ];

    let merged = merge_background(
      &llm,
      "I was born in Colorado.",
      &snippets,
      &BackgroundConfig::default(),
    )
    .await
    .expect("merge");

    assert_eq!(
      merged,
      "I was born in Texas and have ten years of startup experience."
    );

    let calls = llm.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].schema_name, "background_merge");
    assert!(
      calls[0]
        .user
        .contains("CURRENT BACKGROUND:\nI was born in Colorado.")
    );
    assert!(calls[0].user.contains("1. I was born in Texas."));
    assert!(
      calls[0]
        .user
        .contains("2. I have ten years of startup experience.")
    );
    assert!(
      calls[0]
        .system
        .as_deref()
        .is_some_and(|prompt| prompt.contains("first-person"))
    );
  }

  #[tokio::test]
  async fn empty_snippets_skip_the_model_call() {
    let llm = FakeLlm::new();

    let merged = merge_background(&llm, "I am careful.", &[], &BackgroundConfig::default())
      .await
      .expect("merge");

    assert_eq!(merged, "I am careful.");
    assert!(llm.calls().is_empty());
  }

  #[tokio::test]
  async fn rejects_an_empty_merge_result() {
    let llm = FakeLlm::new();
    llm.push("background_merge", json!({ "background": "   " }));

    let error = merge_background(
      &llm,
      "I am careful.",
      &["I am an engineer.".to_string()],
      &BackgroundConfig::default(),
    )
    .await
    .expect_err("empty merge");

    assert!(matches!(error, Error::Fatal(_)));
  }

  #[tokio::test]
  async fn rejects_an_over_long_merge_result() {
    let llm = FakeLlm::new();
    llm.push(
      "background_merge",
      json!({ "background": "I am a very long background." }),
    );
    let config = BackgroundConfig {
      max_chars: 10,
      ..BackgroundConfig::default()
    };

    let error = merge_background(&llm, "", &["I am an engineer.".to_string()], &config)
      .await
      .expect_err("too long");

    assert!(matches!(error, Error::Fatal(_)));
  }
}
