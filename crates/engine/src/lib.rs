//! Core domain types, ports, and orchestration for `memorybank`.
//!
//! This crate is backend- and provider-agnostic: it defines the domain model and
//! the ports (traits) that `store` and `llm` implement.

pub mod error;
pub mod types;

pub use error::Error;

/// Crate identity, used by the workspace smoke test.
pub const CRATE_NAME: &str = "engine";
