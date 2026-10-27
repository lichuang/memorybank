//! retain · inbox, chunking, extraction orchestration, tx2 commit, rescan.

pub mod normalize;

pub use normalize::{NormalizedPayload, RawPayload, RawTurn, TurnSpan, normalize};
