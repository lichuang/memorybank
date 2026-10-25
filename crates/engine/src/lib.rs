//! Core domain types, ports, and orchestration for `memorybank`.
//!
//! This crate is backend- and provider-agnostic: it defines the domain model and
//! the ports (traits) that `store` and `llm` implement.

/// Crate identity, used by the workspace smoke test.
pub const CRATE_NAME: &str = "engine";
