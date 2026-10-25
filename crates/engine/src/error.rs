//! The single error type used across the engine.
//!
//! Classification is part of the contract, not a formatting detail: workers
//! retry only [`Error::Retryable`].
//!
//! - [`Error::Retryable`] — transient failures (LLM timeout, rate limiting, DB
//!   contention); retrying the same work item may succeed.
//! - [`Error::Fatal`] — deterministic failures (invalid model output, data
//!   corruption); retrying cannot help.
//! - [`Error::Config`] — invalid configuration or caller-supplied settings
//!   (e.g. a profile dimension outside 1..=5); the input must be fixed.
//! - [`Error::NotFound`] — an addressed row does not exist.
//! - [`Error::Conflict`] — a uniqueness or state precondition failed (e.g. a
//!   bank name already taken, a chunk already claimed).
//!
//! Adapters (`store`, `llm`) convert their own errors with the constructors
//! below, formatting the underlying cause into the message. The engine never
//! matches on adapter-specific error types.

use thiserror::Error;

use crate::types::ProfileError;

/// The engine-wide error type.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
  /// Transient failure; retrying the same work item may succeed.
  #[error("retryable: {0}")]
  Retryable(String),

  /// Deterministic failure; retrying cannot help.
  #[error("fatal: {0}")]
  Fatal(String),

  /// Invalid configuration or caller-supplied settings.
  #[error("configuration: {0}")]
  Config(String),

  /// An addressed row does not exist: `(entity, id)`.
  #[error("{0} not found: {1}")]
  NotFound(&'static str, String),

  /// A uniqueness or state precondition failed: `(entity, message)`.
  #[error("{0} conflict: {1}")]
  Conflict(&'static str, String),
}

impl Error {
  /// Builds a retryable error from any displayable cause.
  pub fn retryable(message: impl Into<String>) -> Self {
    Self::Retryable(message.into())
  }

  /// Builds a fatal error from any displayable cause.
  pub fn fatal(message: impl Into<String>) -> Self {
    Self::Fatal(message.into())
  }

  /// Builds a configuration error from any displayable cause.
  pub fn config(message: impl Into<String>) -> Self {
    Self::Config(message.into())
  }

  /// Builds a not-found error for `entity` and `id`.
  pub fn not_found(entity: &'static str, id: impl Into<String>) -> Self {
    Self::NotFound(entity, id.into())
  }

  /// Builds a conflict error for `entity` with a human-readable message.
  pub fn conflict(entity: &'static str, message: impl Into<String>) -> Self {
    Self::Conflict(entity, message.into())
  }

  /// Returns `true` only for transient failures that workers may retry.
  pub fn is_retryable(&self) -> bool {
    matches!(self, Self::Retryable(_))
  }
}

impl From<ProfileError> for Error {
  fn from(error: ProfileError) -> Self {
    Self::Config(error.to_string())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  const fn assert_send_sync<T: Send + Sync + 'static>() {}

  #[test]
  fn error_is_send_sync() {
    assert_send_sync::<Error>();
  }

  #[test]
  fn retry_classification_matrix() {
    assert!(Error::retryable("llm timeout").is_retryable());
    assert!(!Error::fatal("invalid JSON for FactExtractionResponse").is_retryable());
    assert!(!Error::config("profile out of range").is_retryable());
    assert!(!Error::not_found("bank", "0190b1c2").is_retryable());
    assert!(!Error::conflict("bank", "name already taken").is_retryable());
  }

  #[test]
  fn display_carries_context() {
    assert_eq!(
      Error::not_found("fact", "f-1").to_string(),
      "fact not found: f-1"
    );
    assert_eq!(
      Error::conflict("chunk", "already claimed").to_string(),
      "chunk conflict: already claimed"
    );
    assert_eq!(
      Error::retryable("db busy").to_string(),
      "retryable: db busy"
    );
  }

  #[test]
  fn profile_errors_are_configuration_errors() {
    let profile = crate::types::BankProfile {
      background: "test".into(),
      skepticism: 0,
      literalism: 3,
      empathy: 3,
      bias_strength: 0.5,
    };
    let profile_error = profile.validate().expect_err("invalid skepticism");

    let error = Error::from(profile_error);

    assert!(matches!(error, Error::Config(_)));
    assert!(!error.is_retryable());
    assert!(error.to_string().contains("skepticism"));
  }
}
