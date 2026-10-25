//! `memorybank` facade.
//!
//! Wires the core [`engine`] to concrete adapters ([`store`], [`llm`]) and exposes
//! the library's public API. Users depend on this crate only; `engine` never
//! depends on an adapter in reverse (see `AGENTS.md` §6).
//!
//! Implementation starts at phase P0 in `design/phase.md`.

pub use engine;
pub use llm;
pub use store;
