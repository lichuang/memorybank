//! Reflect: preference-conditioned reasoning (paper §5).
//!
//! The verbalization of the numeric behavioral profile feeds the system message;
//! the LLM call and opinion persistence build on that.

pub mod llm;
pub mod system_message;
pub mod verbalize;

pub use llm::{ReflectConfig, reflect_llm};
pub use system_message::{SystemMessageConfig, system_message};
pub use verbalize::verbalize;
