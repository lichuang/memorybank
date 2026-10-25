//! Ports (traits) the engine depends on.
//!
//! The engine defines these contracts and the adapters (`store`, `llm`, or
//! user-supplied types) implement them. All ports are object-safe so they can be
//! injected as `Arc<dyn Port>` / `Box<dyn Port>` at runtime.
//!
//! The one deliberate exception is [`LlmExt`], the typed convenience layer over
//! [`Llm`]: it is generic by nature and therefore never used as a trait object.

pub mod clock;
pub mod embedder;
pub mod llm;
pub mod reranker;
pub mod temporal;

pub use clock::{Clock, now_ts};
pub use embedder::Embedder;
pub use llm::{JsonRequest, Llm, LlmExt};
pub use reranker::Reranker;
pub use temporal::TemporalParser;
