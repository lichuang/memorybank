//! Reflect: preference-conditioned reasoning (paper §5).
//!
//! Starts with the verbalization of the numeric behavioral profile; the system
//! message, LLM call, and opinion persistence build on it.

pub mod verbalize;

pub use verbalize::verbalize;
