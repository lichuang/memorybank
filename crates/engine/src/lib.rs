//! Core domain types, ports, and orchestration for `memorybank`.
//!
//! This crate is backend- and provider-agnostic: it defines the domain model and
//! the ports (traits) that `store` and `llm` implement.

pub mod error;
pub mod ports;
pub mod resolve;
pub mod retain;
pub mod testing;
pub mod tokens;
pub mod types;

pub use error::{Error, Result};

/// Crate identity, used by the workspace smoke test.
pub const CRATE_NAME: &str = "engine";
