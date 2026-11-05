//! SQLite adapter.

pub mod store;

pub(crate) mod error;
pub(crate) mod time;

use std::sync::OnceLock;

use engine::{Error, Result};
use sqlx::SqlitePool;
use sqlx::migrate::Migrator;

pub use store::SqliteStore;

/// Fixed embedding dimension of the SQLite vector table (`0001_init.sql`).
///
/// `vec0` requires a static dimension in its DDL, so the schema pins it. The
/// configured `Embedder::dim()` must match; changing to a model with a different
/// dimension requires rebuilding `fact_vec`.
pub const EMBEDDING_DIM: usize = 1536;

/// Registers statically linked SQLite extensions (currently sqlite-vec).
///
/// Must run **before** the pool opens its first connection:
/// `sqlite3_auto_extension` only affects connections created afterwards. It is
/// idempotent, so [`migrate`] and test helpers can call it unconditionally.
#[allow(unsafe_code)] // The only FFI boundary: registering the statically linked extension.
pub fn register_extensions() -> Result<()> {
  static EXTENSIONS: OnceLock<std::result::Result<(), String>> = OnceLock::new();

  let status = EXTENSIONS.get_or_init(|| unsafe {
    // SAFETY: `sqlite3_vec_init` is the entry point compiled into this binary;
    // its C signature matches SQLite's auto-extension callback.
    let entry: unsafe extern "C" fn(
      *mut libsqlite3_sys::sqlite3,
      *mut *mut std::os::raw::c_char,
      *const libsqlite3_sys::sqlite3_api_routines,
    ) -> std::os::raw::c_int = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
    let code = libsqlite3_sys::sqlite3_auto_extension(Some(entry));
    if code == libsqlite3_sys::SQLITE_OK {
      Ok(())
    } else {
      Err(format!(
        "sqlite3_auto_extension(sqlite-vec) failed with code {code}"
      ))
    }
  });

  status.clone().map_err(Error::fatal)
}

/// Verifies that the vector extension is available on this connection.
///
/// Returns a configuration error when the pool was opened before
/// [`register_extensions`] ran; the semantic channel cannot work in that case.
async fn verify_extensions(pool: &SqlitePool) -> Result<()> {
  let version: String = sqlx::query_scalar("SELECT vec_version()")
    .fetch_one(pool)
    .await
    .map_err(|error| {
      Error::config(format!(
        "sqlite-vec is unavailable; call register_extensions() before opening the pool: {error}"
      ))
    })?;

  tracing::debug!(version, "sqlite-vec loaded");
  Ok(())
}

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
  // The initial migration creates the sqlite-vec table, so the extension must
  // already be loaded on this connection.
  verify_extensions(pool).await?;

  MIGRATOR
    .run(pool)
    .await
    .map_err(|error| Error::fatal(format!("sqlite migration failed: {error}")))
}
