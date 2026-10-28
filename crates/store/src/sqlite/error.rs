//! SQLite errors → [`engine::Error`] classification.
//!
//! Kept in the adapter because it depends on `sqlx` and on SQLite result codes;
//! the engine must not know either.

use engine::Error;

/// Maps a write failure: unique violations become [`Error::Conflict`], SQLite
/// contention becomes [`Error::Retryable`], everything else is fatal.
pub(crate) fn map_write(entity: &'static str, error: sqlx::Error) -> Error {
  if let sqlx::Error::Database(database) = &error {
    if database.is_unique_violation() {
      return Error::conflict(entity, error.to_string());
    }
    if is_contention(database.as_ref()) {
      return Error::retryable(error.to_string());
    }
  }
  Error::fatal(format!("{entity} write failed: {error}"))
}

/// Maps a read failure the same way, minus the uniqueness case.
pub(crate) fn map_read(entity: &'static str, error: sqlx::Error) -> Error {
  if let sqlx::Error::Database(database) = &error
    && is_contention(database.as_ref())
  {
    return Error::retryable(error.to_string());
  }
  Error::fatal(format!("{entity} read failed: {error}"))
}

/// `SQLITE_BUSY` (5) and `SQLITE_LOCKED` (6), including extended result codes.
fn is_contention(database: &dyn sqlx::error::DatabaseError) -> bool {
  database
    .code()
    .and_then(|code| code.parse::<u32>().ok())
    .is_some_and(|code| matches!(code & 0xff, 5 | 6))
}
