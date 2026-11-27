//! Reflect: preference-conditioned reasoning (paper §5).
//!
//! The verbalization of the numeric behavioral profile feeds the system message;
//! the LLM call and opinion persistence build on that.

pub mod llm;
pub mod opinion;
pub mod result;
pub mod system_message;
pub mod verbalize;

pub use llm::{ReflectConfig, reflect_llm};
pub use opinion::{bind_known_entities, build_opinion, prepare_opinion};
pub use result::ReflectResult;
pub use system_message::{SystemMessageConfig, system_message};
pub use verbalize::verbalize;
