//! SQLite storage adapter for `memorybank`.

pub mod sqlite;
pub mod testing;

pub use sqlite::migrate;

/// Crate identity, used by the workspace smoke test.
pub const CRATE_NAME: &str = "store";
