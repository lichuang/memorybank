//! SQLite implementation of the engine [`Store`] port.
//!
//! Implemented incrementally alongside the P1 tasks: bank/source writes first,
//! the inbox and retrieval methods follow. Unimplemented methods return a
//! [`Error::Fatal`] instead of panicking, so an accidental call is visible as an
//! error rather than a crash.

use std::time::Duration;

use async_trait::async_trait;
use engine::ports::{
  ChunkCommit, ClaimedChunk, EntityCandidate, Neighbor, NewChunk, NewEntity, NewObservation,
  NewOpinion, Store, WorkLists,
};
use engine::types::{
  Bank, BankId, BankProfile, Chunk, ChunkId, ChunkStatus, EntityId, Fact, FactId, Source, SourceId,
  SourceStatus, TimeRange, Timestamp,
};
use engine::{Error, Result};
use sqlx::SqlitePool;

/// A [`Store`] backed by a SQLite connection pool.
#[derive(Debug, Clone)]
pub struct SqliteStore {
  pool: SqlitePool,
}

impl SqliteStore {
  /// Wraps an already-migrated pool.
  pub fn new(pool: SqlitePool) -> Self {
    Self { pool }
  }

  /// Returns the underlying pool (used by adapter tests and maintenance tools).
  pub fn pool(&self) -> &SqlitePool {
    &self.pool
  }
}

#[async_trait]
impl Store for SqliteStore {
  async fn create_bank(&self, bank: Bank) -> Result<()> {
    sqlx::query(
      "INSERT INTO bank (id, name, background, skepticism, literalism, empathy, bias_strength, \
       created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(bank.id.to_string())
    .bind(&bank.name)
    .bind(&bank.profile.background)
    .bind(i64::from(bank.profile.skepticism))
    .bind(i64::from(bank.profile.literalism))
    .bind(i64::from(bank.profile.empathy))
    .bind(f64::from(bank.profile.bias_strength))
    .bind(epoch_millis(bank.created_at))
    .bind(epoch_millis(bank.updated_at))
    .execute(&self.pool)
    .await
    .map_err(|error| map_write_error("bank", error))?;

    Ok(())
  }

  async fn load_bank(&self, id: BankId) -> Result<Bank> {
    type Row = (String, String, String, i64, i64, i64, f64, i64, i64);

    let row = sqlx::query_as::<_, Row>(
      "SELECT id, name, background, skepticism, literalism, empathy, bias_strength, \
       created_at, updated_at FROM bank WHERE id = ?",
    )
    .bind(id.to_string())
    .fetch_optional(&self.pool)
    .await
    .map_err(|error| map_read_error("bank", error))?
    .ok_or_else(|| Error::not_found("bank", id.to_string()))?;

    let (
      id_text,
      name,
      background,
      skepticism,
      literalism,
      empathy,
      bias_strength,
      created_at,
      updated_at,
    ) = row;

    let bank_id = BankId::parse(&id_text)
      .map_err(|error| Error::fatal(format!("corrupt bank id `{id_text}`: {error}")))?;

    let profile = BankProfile {
      background,
      skepticism: level("skepticism", skepticism)?,
      literalism: level("literalism", literalism)?,
      empathy: level("empathy", empathy)?,
      bias_strength: bias_strength as f32,
    };

    Ok(Bank {
      id: bank_id,
      name,
      profile,
      created_at: timestamp(created_at)?,
      updated_at: timestamp(updated_at)?,
    })
  }

  async fn create_source(&self, source: Source) -> Result<()> {
    sqlx::query(
      "INSERT INTO source (id, bank_id, title, payload_text, status, created_at, updated_at) \
       VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(source.id.to_string())
    .bind(source.bank_id.to_string())
    .bind(source.title.as_deref())
    .bind(&source.payload_text)
    .bind(source_status(source.status))
    .bind(epoch_millis(source.created_at))
    .bind(epoch_millis(source.updated_at))
    .execute(&self.pool)
    .await
    .map_err(|error| map_write_error("source", error))?;

    Ok(())
  }

  // —— Not implemented yet (P1-4 onward) ——

  async fn append_chunks(&self, source: SourceId, chunks: Vec<NewChunk>) -> Result<usize> {
    if chunks.iter().any(|chunk| chunk.source_id != source) {
      return Err(Error::config(
        "append_chunks: every chunk must belong to the given source",
      ));
    }

    // Option A: derive `source.updated_at` from the batch instead of taking a
    // clock reading. An empty batch (nothing to derive from) keeps the old value.
    let updated_at = chunks.iter().map(|chunk| chunk.created_at).max();
    let mut tx = self
      .pool
      .begin()
      .await
      .map_err(|error| map_write_error("chunk", error))?;

    // The source must exist and own the batch's bank (schema §3.3).
    let stored_bank: Option<String> = sqlx::query_scalar("SELECT bank_id FROM source WHERE id = ?")
      .bind(source.to_string())
      .fetch_optional(&mut *tx)
      .await
      .map_err(|error| map_read_error("source", error))?;
    let stored_bank = stored_bank.ok_or_else(|| Error::not_found("source", source.to_string()))?;
    let bank_id = BankId::parse(&stored_bank)
      .map_err(|error| Error::fatal(format!("corrupt bank id `{stored_bank}`: {error}")))?;

    if chunks.iter().any(|chunk| chunk.bank_id != bank_id) {
      // Dropping `tx` rolls back (nothing has been written yet).
      return Err(Error::config(
        "append_chunks: every chunk must belong to the source's bank",
      ));
    }

    let mut inserted = 0_usize;
    for chunk in &chunks {
      let result = sqlx::query(
        "INSERT INTO chunk (id, bank_id, source_id, chunk_index, text, content_hash, status, \
         time_start, time_end, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?) \
         ON CONFLICT(bank_id, content_hash) DO NOTHING",
      )
      .bind(chunk.id.to_string())
      .bind(chunk.bank_id.to_string())
      .bind(chunk.source_id.to_string())
      .bind(i64::from(chunk.chunk_index))
      .bind(&chunk.text)
      .bind(&chunk.content_hash)
      .bind(chunk.time_start.map(epoch_millis))
      .bind(chunk.time_end.map(epoch_millis))
      .bind(epoch_millis(chunk.created_at))
      .bind(epoch_millis(chunk.created_at))
      .execute(&mut *tx)
      .await
      .map_err(|error| map_write_error("chunk", error))?;
      inserted += result.rows_affected() as usize;
    }

    let updated = sqlx::query(
      "UPDATE source SET status = 'chunked', updated_at = COALESCE(?, updated_at) WHERE id = ?",
    )
    .bind(updated_at.map(epoch_millis))
    .bind(source.to_string())
    .execute(&mut *tx)
    .await
    .map_err(|error| map_write_error("source", error))?;

    if updated.rows_affected() == 0 {
      // Defensive: the source was read above inside this transaction.
      return Err(Error::not_found("source", source.to_string()));
    }

    tx.commit()
      .await
      .map_err(|error| map_write_error("source", error))?;

    Ok(inserted)
  }

  async fn claim_chunk(
    &self,
    bank: BankId,
    now: Timestamp,
    stale_after: Duration,
  ) -> Result<Option<ClaimedChunk>> {
    let now_millis = epoch_millis(now);
    let stale_before = now_millis.saturating_sub(duration_millis(stale_after));

    let row = sqlx::query_as::<_, ChunkRow>(
      "UPDATE chunk SET status = 'extracting', updated_at = MAX(?, updated_at + 1) \
       WHERE id = ( \
         SELECT id FROM chunk \
         WHERE bank_id = ? \
           AND (status = 'pending' OR (status = 'extracting' AND updated_at < ?)) \
         ORDER BY created_at, id LIMIT 1 \
       ) \
       RETURNING id, bank_id, source_id, chunk_index, text, content_hash, status, \
                 time_start, time_end, created_at, updated_at",
    )
    .bind(now_millis)
    .bind(bank.to_string())
    .bind(stale_before)
    .fetch_optional(&self.pool)
    .await
    .map_err(|error| map_write_error("chunk", error))?;

    row
      .map(chunk_from_row)
      .transpose()
      .map(|claimed| claimed.map(|chunk| ClaimedChunk { chunk }))
  }

  async fn release_chunk(&self, _chunk: ChunkId) -> Result<()> {
    Err(not_implemented("release_chunk"))
  }

  async fn commit_chunk(&self, _commit: ChunkCommit) -> Result<()> {
    Err(not_implemented("commit_chunk"))
  }

  async fn search_vector(
    &self,
    _bank: BankId,
    _query: &[f32],
    _k: usize,
  ) -> Result<Vec<(FactId, f32)>> {
    Err(not_implemented("search_vector"))
  }

  async fn search_bm25(
    &self,
    _bank: BankId,
    _query: &str,
    _k: usize,
  ) -> Result<Vec<(FactId, f32)>> {
    Err(not_implemented("search_bm25"))
  }

  async fn neighbors(&self, _bank: BankId, _from: FactId) -> Result<Vec<Neighbor>> {
    Err(not_implemented("neighbors"))
  }

  async fn facts_in_range(&self, _bank: BankId, _range: TimeRange) -> Result<Vec<FactId>> {
    Err(not_implemented("facts_in_range"))
  }

  async fn fetch_facts(&self, _ids: &[FactId]) -> Result<Vec<Fact>> {
    Err(not_implemented("fetch_facts"))
  }

  async fn increment_access(&self, _ids: &[FactId]) -> Result<()> {
    Err(not_implemented("increment_access"))
  }

  async fn ensure_entity(&self, _new: NewEntity) -> Result<(EntityId, bool)> {
    Err(not_implemented("ensure_entity"))
  }

  async fn entity_candidates(&self, _bank: BankId, _mention: &str) -> Result<Vec<EntityCandidate>> {
    Err(not_implemented("entity_candidates"))
  }

  async fn insert_opinion(&self, _opinion: NewOpinion) -> Result<()> {
    Err(not_implemented("insert_opinion"))
  }

  async fn insert_observations(&self, _observations: Vec<NewObservation>) -> Result<()> {
    Err(not_implemented("insert_observations"))
  }

  async fn observations_of(
    &self,
    _bank: BankId,
    _entity: EntityId,
    _limit: usize,
  ) -> Result<Vec<Fact>> {
    Err(not_implemented("observations_of"))
  }

  async fn pending_work(
    &self,
    _bank: BankId,
    _now: Timestamp,
    _stale_after: Duration,
  ) -> Result<WorkLists> {
    Err(not_implemented("pending_work"))
  }

  async fn update_embeddings(&self, _embeddings: &[(FactId, Vec<f32>)]) -> Result<usize> {
    Err(not_implemented("update_embeddings"))
  }

  async fn gc_observations(
    &self,
    _bank: BankId,
    _now: Timestamp,
    _max_age: Duration,
    _per_entity_cap: usize,
  ) -> Result<usize> {
    Err(not_implemented("gc_observations"))
  }
}

fn not_implemented(method: &str) -> Error {
  Error::fatal(format!("SqliteStore::{method} is not implemented yet"))
}

fn epoch_millis(timestamp: Timestamp) -> i64 {
  timestamp.timestamp_millis()
}

fn duration_millis(duration: Duration) -> i64 {
  i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

type ChunkRow = (
  String,
  String,
  String,
  i64,
  String,
  String,
  String,
  Option<i64>,
  Option<i64>,
  i64,
  i64,
);

fn chunk_from_row(row: ChunkRow) -> Result<Chunk> {
  let (
    id,
    bank_id,
    source_id,
    chunk_index,
    text,
    content_hash,
    status,
    time_start,
    time_end,
    created_at,
    updated_at,
  ) = row;

  Ok(Chunk {
    id: ChunkId::parse(&id)
      .map_err(|error| Error::fatal(format!("corrupt chunk id `{id}`: {error}")))?,
    bank_id: BankId::parse(&bank_id)
      .map_err(|error| Error::fatal(format!("corrupt bank id `{bank_id}`: {error}")))?,
    source_id: SourceId::parse(&source_id)
      .map_err(|error| Error::fatal(format!("corrupt source id `{source_id}`: {error}")))?,
    chunk_index: u32::try_from(chunk_index)
      .map_err(|_| Error::fatal(format!("corrupt chunk index `{chunk_index}`")))?,
    text,
    content_hash,
    status: chunk_status(&status)?,
    time_start: time_start.map(timestamp).transpose()?,
    time_end: time_end.map(timestamp).transpose()?,
    created_at: timestamp(created_at)?,
    updated_at: timestamp(updated_at)?,
  })
}

fn chunk_status(value: &str) -> Result<ChunkStatus> {
  match value {
    "pending" => Ok(ChunkStatus::Pending),
    "extracting" => Ok(ChunkStatus::Extracting),
    "done" => Ok(ChunkStatus::Done),
    "failed" => Ok(ChunkStatus::Failed),
    other => Err(Error::fatal(format!("corrupt chunk status `{other}`"))),
  }
}

fn timestamp(millis: i64) -> Result<Timestamp> {
  Timestamp::from_timestamp_millis(millis)
    .ok_or_else(|| Error::fatal(format!("corrupt timestamp `{millis}` in database")))
}

fn level(field: &'static str, value: i64) -> Result<u8> {
  u8::try_from(value)
    .ok()
    .filter(|value| (1..=5).contains(value))
    .ok_or_else(|| Error::fatal(format!("corrupt profile level `{field}` = {value}")))
}

fn source_status(status: SourceStatus) -> &'static str {
  match status {
    SourceStatus::Received => "received",
    SourceStatus::Chunked => "chunked",
  }
}

/// Maps write failures: unique violations become [`Error::Conflict`], SQLite
/// contention becomes [`Error::Retryable`], everything else is fatal.
fn map_write_error(entity: &'static str, error: sqlx::Error) -> Error {
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

/// Maps read failures the same way, minus the uniqueness case.
fn map_read_error(entity: &'static str, error: sqlx::Error) -> Error {
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
