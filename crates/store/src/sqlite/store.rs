//! SQLite implementation of the engine [`Store`] port.
//!
//! Implemented incrementally alongside the P1 tasks: bank/source writes first,
//! the inbox and retrieval methods follow. Unimplemented methods return a
//! [`Error::Fatal`] instead of panicking, so an accidental call is visible as an
//! error rather than a crash.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use async_trait::async_trait;
use engine::ports::{
  ChunkCommit, ClaimedChunk, EntityCandidate, Neighbor, NewChunk, NewEntity, NewObservation,
  NewOpinion, Store, WorkLists,
};
use engine::types::{
  Bank, BankId, BankProfile, Chunk, ChunkId, ChunkStatus, Entity, EntityId, EntityKind, Fact,
  FactId, Source, SourceId, TimeRange, Timestamp, from_epoch_millis, to_epoch_millis,
};
use engine::{Error, Result};
use sqlx::{QueryBuilder, Sqlite, SqlitePool};

use super::error::{map_read, map_write};
use super::time::duration_millis;

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
    .bind(to_epoch_millis(bank.created_at))
    .bind(to_epoch_millis(bank.updated_at))
    .execute(&self.pool)
    .await
    .map_err(|error| map_write("bank", error))?;

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
    .map_err(|error| map_read("bank", error))?
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
    .bind(source.status.as_str())
    .bind(to_epoch_millis(source.created_at))
    .bind(to_epoch_millis(source.updated_at))
    .execute(&self.pool)
    .await
    .map_err(|error| map_write("source", error))?;

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
      .map_err(|error| map_write("chunk", error))?;

    // The source must exist and own the batch's bank (schema §3.3).
    let stored_bank: Option<String> = sqlx::query_scalar("SELECT bank_id FROM source WHERE id = ?")
      .bind(source.to_string())
      .fetch_optional(&mut *tx)
      .await
      .map_err(|error| map_read("source", error))?;
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
      .bind(chunk.time_start.map(to_epoch_millis))
      .bind(chunk.time_end.map(to_epoch_millis))
      .bind(to_epoch_millis(chunk.created_at))
      .bind(to_epoch_millis(chunk.created_at))
      .execute(&mut *tx)
      .await
      .map_err(|error| map_write("chunk", error))?;
      inserted += result.rows_affected() as usize;
    }

    let updated = sqlx::query(
      "UPDATE source SET status = 'chunked', updated_at = COALESCE(?, updated_at) WHERE id = ?",
    )
    .bind(updated_at.map(to_epoch_millis))
    .bind(source.to_string())
    .execute(&mut *tx)
    .await
    .map_err(|error| map_write("source", error))?;

    if updated.rows_affected() == 0 {
      // Defensive: the source was read above inside this transaction.
      return Err(Error::not_found("source", source.to_string()));
    }

    tx.commit()
      .await
      .map_err(|error| map_write("source", error))?;

    Ok(inserted)
  }

  async fn claim_chunk(
    &self,
    bank: BankId,
    now: Timestamp,
    stale_after: Duration,
  ) -> Result<Option<ClaimedChunk>> {
    let now_millis = to_epoch_millis(now);
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
    .map_err(|error| map_write("chunk", error))?;

    row
      .map(chunk_from_row)
      .transpose()
      .map(|claimed| claimed.map(|chunk| ClaimedChunk { chunk }))
  }

  async fn release_chunk(&self, chunk: ChunkId, now: Timestamp) -> Result<()> {
    let released = sqlx::query(
      "UPDATE chunk SET status = 'pending', updated_at = ? WHERE id = ? AND status = 'extracting'",
    )
    .bind(to_epoch_millis(now))
    .bind(chunk.to_string())
    .execute(&self.pool)
    .await
    .map_err(|error| map_write("chunk", error))?;

    if released.rows_affected() > 0 {
      return Ok(());
    }

    // Nothing was updated: either the row is gone, or it is not `extracting`
    // (already released, done, or dead-lettered). Never resurrect those states.
    let status: Option<String> = sqlx::query_scalar("SELECT status FROM chunk WHERE id = ?")
      .bind(chunk.to_string())
      .fetch_optional(&self.pool)
      .await
      .map_err(|error| map_read("chunk", error))?;

    match status {
      Some(_) => Ok(()),
      None => Err(Error::not_found("chunk", chunk.to_string())),
    }
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

  async fn ensure_entity(&self, new: NewEntity) -> Result<(EntityId, bool)> {
    if new.canonical_name.trim().is_empty() {
      return Err(Error::config(
        "ensure_entity: canonical_name must not be empty",
      ));
    }

    let inserted = sqlx::query(
      "INSERT INTO entity (id, bank_id, canonical_name, entity_type, mention_count, \
       first_seen, last_seen, attrs, created_at, updated_at) \
       VALUES (?, ?, ?, ?, 0, NULL, NULL, NULL, ?, ?) \
       ON CONFLICT(bank_id, canonical_name) DO NOTHING",
    )
    .bind(new.id.to_string())
    .bind(new.bank_id.to_string())
    .bind(&new.canonical_name)
    .bind(new.entity_type.as_str())
    .bind(to_epoch_millis(new.created_at))
    .bind(to_epoch_millis(new.created_at))
    .execute(&self.pool)
    .await
    .map_err(|error| map_write("entity", error))?;

    if inserted.rows_affected() == 1 {
      return Ok((new.id, true));
    }

    let existing: Option<String> =
      sqlx::query_scalar("SELECT id FROM entity WHERE bank_id = ? AND canonical_name = ?")
        .bind(new.bank_id.to_string())
        .bind(&new.canonical_name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| map_read("entity", error))?;

    match existing {
      Some(id) => {
        let id = EntityId::parse(&id)
          .map_err(|error| Error::fatal(format!("corrupt entity id `{id}`: {error}")))?;
        Ok((id, false))
      }
      None => Err(Error::fatal(
        "entity conflict reported but no existing row was found",
      )),
    }
  }

  async fn entity_candidates(
    &self,
    bank: BankId,
    mention: &str,
    limit: usize,
  ) -> Result<Vec<EntityCandidate>> {
    let limit = limit.max(1);
    let bank_text = bank.to_string();

    // Exact canonical-name and variant matches first, then the bank's top
    // entities by mention_count so fuzzy scoring still sees new spellings.
    let mut rows: Vec<EntityRow> = sqlx::query_as::<_, EntityRow>(
      "SELECT e.id, e.bank_id, e.canonical_name, e.entity_type, e.mention_count, \
              e.first_seen, e.last_seen, e.attrs, e.created_at, e.updated_at \
       FROM entity e \
       WHERE e.bank_id = ? \
         AND (e.canonical_name = ? OR EXISTS ( \
           SELECT 1 FROM fact_entity fe \
           WHERE fe.entity_id = e.id AND fe.bank_id = e.bank_id \
             AND lower(trim(fe.mention_text)) = ?)) \
       ORDER BY e.mention_count DESC, e.id LIMIT ?",
    )
    .bind(&bank_text)
    .bind(mention)
    .bind(mention)
    .bind(limit as i64)
    .fetch_all(&self.pool)
    .await
    .map_err(|error| map_read("entity", error))?;

    let top: Vec<EntityRow> = sqlx::query_as::<_, EntityRow>(
      "SELECT id, bank_id, canonical_name, entity_type, mention_count, first_seen, last_seen, \
              attrs, created_at, updated_at \
       FROM entity WHERE bank_id = ? ORDER BY mention_count DESC, id LIMIT ?",
    )
    .bind(&bank_text)
    .bind(limit as i64)
    .fetch_all(&self.pool)
    .await
    .map_err(|error| map_read("entity", error))?;

    let mut seen: HashSet<String> = HashSet::new();
    rows.retain(|row| seen.insert(row.0.clone()));
    for row in top {
      if rows.len() >= limit {
        break;
      }
      if seen.insert(row.0.clone()) {
        rows.push(row);
      }
    }
    rows.truncate(limit);

    if rows.is_empty() {
      return Ok(Vec::new());
    }

    let ids: Vec<String> = rows.iter().map(|row| row.0.clone()).collect();

    let mut variants_query = QueryBuilder::<Sqlite>::new(
      "SELECT entity_id, mention_text FROM fact_entity WHERE bank_id = ",
    );
    variants_query.push_bind(&bank_text);
    variants_query.push(" AND entity_id IN (");
    let mut separated = variants_query.separated(", ");
    for id in &ids {
      separated.push_bind(id.clone());
    }
    variants_query.push(") AND mention_text IS NOT NULL");
    let variant_rows: Vec<(String, String)> = variants_query
      .build_query_as()
      .fetch_all(&self.pool)
      .await
      .map_err(|error| map_read("entity", error))?;

    let mut co_query = QueryBuilder::<Sqlite>::new(
      "SELECT fe1.entity_id, fe2.entity_id FROM fact_entity fe1 \
       JOIN fact_entity fe2 ON fe2.fact_id = fe1.fact_id AND fe2.entity_id <> fe1.entity_id \
       WHERE fe1.bank_id = ",
    );
    co_query.push_bind(&bank_text);
    co_query.push(" AND fe1.entity_id IN (");
    let mut separated = co_query.separated(", ");
    for id in &ids {
      separated.push_bind(id.clone());
    }
    // Bounded sample; fact ids are UUIDv7, so descending order is recent-first.
    co_query.push(") ORDER BY fe1.fact_id DESC LIMIT 20000");
    let co_rows: Vec<(String, String)> = co_query
      .build_query_as()
      .fetch_all(&self.pool)
      .await
      .map_err(|error| map_read("entity", error))?;

    let mut variants: HashMap<String, Vec<String>> = HashMap::new();
    for (entity, text) in variant_rows {
      variants.entry(entity).or_default().push(text);
    }
    for values in variants.values_mut() {
      values.sort();
      values.dedup();
    }

    let mut co_mentioned: HashMap<String, Vec<String>> = HashMap::new();
    for (entity, other) in co_rows {
      co_mentioned.entry(entity).or_default().push(other);
    }

    let mut candidates = Vec::with_capacity(rows.len());
    for row in rows {
      let id_text = row.0.clone();
      let entity = entity_from_row(row)?;
      let last_mention_at = entity.last_seen;
      let co_mentioned = co_mentioned
        .remove(&id_text)
        .unwrap_or_default()
        .into_iter()
        .map(|value| {
          EntityId::parse(&value)
            .map_err(|error| Error::fatal(format!("corrupt entity id `{value}`: {error}")))
        })
        .collect::<Result<Vec<_>>>()?;

      candidates.push(EntityCandidate {
        entity,
        variants: variants.remove(&id_text).unwrap_or_default(),
        co_mentioned,
        last_mention_at,
      });
    }

    Ok(candidates)
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
    status: status
      .parse::<ChunkStatus>()
      .map_err(|error| Error::fatal(error.to_string()))?,
    time_start: time_start.map(timestamp).transpose()?,
    time_end: time_end.map(timestamp).transpose()?,
    created_at: timestamp(created_at)?,
    updated_at: timestamp(updated_at)?,
  })
}

type EntityRow = (
  String,
  String,
  String,
  String,
  i64,
  Option<i64>,
  Option<i64>,
  Option<String>,
  i64,
  i64,
);

fn entity_from_row(row: EntityRow) -> Result<Entity> {
  let (
    id,
    bank_id,
    canonical_name,
    entity_type,
    mention_count,
    first_seen,
    last_seen,
    attrs,
    created_at,
    updated_at,
  ) = row;

  Ok(Entity {
    id: EntityId::parse(&id)
      .map_err(|error| Error::fatal(format!("corrupt entity id `{id}`: {error}")))?,
    bank_id: BankId::parse(&bank_id)
      .map_err(|error| Error::fatal(format!("corrupt bank id `{bank_id}`: {error}")))?,
    canonical_name,
    entity_type: entity_type
      .parse::<EntityKind>()
      .map_err(|error| Error::fatal(error.to_string()))?,
    mention_count: u32::try_from(mention_count)
      .map_err(|_| Error::fatal(format!("corrupt mention count `{mention_count}`")))?,
    first_seen: first_seen.map(timestamp).transpose()?,
    last_seen: last_seen.map(timestamp).transpose()?,
    attrs: attrs
      .map(|text| {
        serde_json::from_str(&text)
          .map_err(|error| Error::fatal(format!("corrupt entity attrs: {error}")))
      })
      .transpose()?,
    created_at: timestamp(created_at)?,
    updated_at: timestamp(updated_at)?,
  })
}

fn timestamp(millis: i64) -> Result<Timestamp> {
  from_epoch_millis(millis)
    .ok_or_else(|| Error::fatal(format!("corrupt timestamp `{millis}` in database")))
}

fn level(field: &'static str, value: i64) -> Result<u8> {
  u8::try_from(value)
    .ok()
    .filter(|value| (1..=5).contains(value))
    .ok_or_else(|| Error::fatal(format!("corrupt profile level `{field}` = {value}")))
}
