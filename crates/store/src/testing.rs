//! Test helpers shared by store tests and integration tests.

use engine::Result;
use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;

/// Opens a fresh, migrated, in-memory SQLite database.
///
/// `max_connections(1)` is deliberate: every SQLite connection to
/// `sqlite::memory:` gets its **own** private database, so a pool with more than
/// one connection would silently split the schema. One connection keeps the
/// database intact; concurrent callers queue on the pool. Each call returns an
/// isolated database, so tests never share state.
pub async fn memory_pool() -> Result<SqlitePool> {
  let pool = SqlitePoolOptions::new()
    .max_connections(1)
    .idle_timeout(None)
    .max_lifetime(None)
    .connect("sqlite::memory:")
    .await
    .map_err(|error| engine::Error::fatal(format!("open in-memory sqlite: {error}")))?;

  crate::sqlite::migrate(&pool).await?;

  Ok(pool)
}
