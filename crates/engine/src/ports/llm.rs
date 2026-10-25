//! Language-model port.
//!
//! [`Llm`] is the dyn-safe transport: one structured call, raw JSON back. The
//! typed layer lives on [`LlmExt`] because a generic method on [`Llm`] would make
//! the trait non-object-safe and rule out `Box<dyn Llm>` / `Arc<dyn Llm>`.

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{Error, Result};

/// A structured-output request.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonRequest {
  /// Optional system prompt.
  pub system: Option<String>,
  /// User prompt with the rendered task and context.
  pub user: String,
  /// Stable name of the expected response schema; also used as the provider's
  /// response-format name.
  pub schema_name: &'static str,
  /// JSON Schema generated from the response DTO (see `schemars::schema_for!`).
  pub schema: Value,
  /// Sampling temperature.
  pub temperature: f32,
  /// Maximum number of tokens to generate.
  pub max_tokens: u32,
}

/// Model transport: sends one structured-output request and returns raw JSON.
#[async_trait]
pub trait Llm: Send + Sync {
  /// Sends `req` and returns the response parsed as JSON.
  ///
  /// Errors: [`Error::Retryable`] for timeouts and rate limiting,
  /// [`Error::Fatal`] for provider-rejected requests or unrecoverable output.
  async fn complete_json_raw(&self, req: JsonRequest) -> Result<Value>;

  /// Model identifier, recorded for observability and cutover checks.
  fn model(&self) -> &str;
}

/// Typed convenience layer over [`Llm`].
///
/// Callers import this trait to call `complete_json::<T>(req)` on any [`Llm`],
/// including trait objects (`dyn Llm` satisfies the blanket impl because the
/// bound is `Llm + ?Sized`).
#[async_trait]
pub trait LlmExt: Llm {
  /// Sends `req` and deserializes the response into `T`.
  ///
  /// A response that does not match `T` is a contract violation, so the
  /// resulting error is [`Error::Fatal`] (retrying the same prompt will not fix
  /// it).
  async fn complete_json<T: DeserializeOwned>(&self, req: JsonRequest) -> Result<T>;
}

#[async_trait]
impl<T: Llm + ?Sized> LlmExt for T {
  async fn complete_json<U: DeserializeOwned>(&self, req: JsonRequest) -> Result<U> {
    let schema_name = req.schema_name;
    let value = self.complete_json_raw(req).await?;
    serde_json::from_value(value)
      .map_err(|error| Error::fatal(format!("invalid {schema_name} output: {error}")))
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::types::{FactExtractionResponse, ReflectResponse};

  /// Minimal adapter: returns an empty extraction response.
  struct StubLlm;

  #[async_trait]
  impl Llm for StubLlm {
    async fn complete_json_raw(&self, _req: JsonRequest) -> Result<Value> {
      Ok(json!({ "facts": [] }))
    }

    fn model(&self) -> &str {
      "stub"
    }
  }

  fn request() -> JsonRequest {
    JsonRequest {
      system: None,
      user: "extract facts".into(),
      schema_name: "fact_extraction",
      schema: json!({ "type": "object" }),
      temperature: 0.0,
      max_tokens: 512,
    }
  }

  #[tokio::test]
  async fn dyn_llm_supports_raw_and_typed_calls() {
    let llm: Box<dyn Llm> = Box::new(StubLlm);

    let raw = llm.complete_json_raw(request()).await.expect("raw call");
    assert_eq!(raw, json!({ "facts": [] }));

    let typed: FactExtractionResponse = llm.complete_json(request()).await.expect("typed call");
    assert!(typed.facts.is_empty());

    assert_eq!(llm.model(), "stub");
  }

  #[tokio::test]
  async fn schema_mismatch_is_fatal() {
    let llm: Box<dyn Llm> = Box::new(StubLlm);

    // ReflectResponse requires `answer`; the stub returns `{ "facts": [] }`.
    let error = llm
      .complete_json::<ReflectResponse>(request())
      .await
      .expect_err("schema mismatch");

    assert!(matches!(error, Error::Fatal(_)));
    assert!(!error.is_retryable());
    assert!(error.to_string().contains("fact_extraction"));
  }
}
