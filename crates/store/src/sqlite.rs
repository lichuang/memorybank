//! SQLite adapter.

use engine::{Error, Result};
use sqlx::SqlitePool;
use sqlx::migrate::Migrator;

/// Migrations are embedded at compile time from `migrations/sqlite/`, so the
/// binary never depends on SQL files being present at runtime or on the working
/// directory. The directory is appended to, never edited in place.
static MIGRATOR: Migrator = sqlx::migrate!("migrations/sqlite");

/// Applies all pending SQLite migrations to `pool`.
///
/// Running this repeatedly is a no-op for already-applied versions; sqlx records
/// each version and its checksum in `_sqlx_migrations`.
///
/// # Errors
///
/// Returns [`Error::Fatal`] when a migration fails or a previously applied file
/// was modified (checksum mismatch) — neither is retryable.
pub async fn migrate(pool: &SqlitePool) -> Result<()> {
  MIGRATOR
    .run(pool)
    .await
    .map_err(|error| Error::fatal(format!("sqlite migration failed: {error}")))
}
