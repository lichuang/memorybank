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
  ChunkCommit, ClaimedChunk, EntityCandidate, EntityStatUpdate, Neighbor, NewChunk, NewEntity,
  NewObservation, NewOpinion, SourceProgress, Store, WorkLists,
};
use engine::types::{
  Bank, BankId, BankProfile, Chunk, ChunkId, ChunkStatus, Entity, EntityId, EntityKind, Fact,
  FactId, FactType, OccurrenceInterval, Source, SourceId, SourceStatus, TimeRange, Timestamp,
  from_epoch_millis, to_epoch_millis,
};
use engine::{Error, Result};
use sqlx::{QueryBuilder, Sqlite, SqlitePool, Transaction};
use tracing::warn;

use super::EMBEDDING_DIM;
use super::error::{map_read, map_write};
use super::time::duration_millis;

/// A [`Store`] backed by a SQLite connection pool.
#[derive(Debug, Clone)]
pub struct SqliteStore {
  pool: SqlitePool,
}

impl SqliteStore {
  /// Wraps an already-migrated pool.
  ///
  /// Call [`crate::sqlite::register_extensions`] **before** opening the pool:
  /// sqlite-vec is loaded through `sqlite3_auto_extension`, which only applies
  /// to connections created after registration.
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

  async fn commit_chunk(&self, commit: ChunkCommit) -> Result<()> {
    let mut tx = self
      .pool
      .begin()
      .await
      .map_err(|error| map_write("chunk commit", error))?;

    match commit_chunk_tx(&mut tx, &commit).await {
      Ok(()) => tx
        .commit()
        .await
        .map_err(|error| map_write("chunk commit", error)),
      Err(error) => {
        let _ = tx.rollback().await;
        Err(error)
      }
    }
  }

  async fn search_vector(
    &self,
    bank: BankId,
    query: &[f32],
    k: usize,
  ) -> Result<Vec<(FactId, f32)>> {
    if k == 0 {
      return Ok(Vec::new());
    }
    if query.len() != EMBEDDING_DIM {
      return Err(Error::config(format!(
        "embedding dimension mismatch: query has {} dimensions, the schema fixes {EMBEDDING_DIM}",
        query.len()
      )));
    }

    // `+1` improves tie handling at the boundary: vec0 may return equal
    // distances in any order, so over-fetch one row, sort by (score, id), then
    // truncate.
    let fetch_k = k.saturating_add(1);

    let rows: Vec<(String, f64)> = sqlx::query_as(
      "SELECT f.id, v.distance FROM fact_vec v JOIN fact f ON f.rowid = v.rowid \
       WHERE v.embedding MATCH ? AND k = ? AND v.bank_id = ?",
    )
    .bind(encode_embedding(query))
    .bind(i64::try_from(fetch_k).unwrap_or(i64::MAX))
    .bind(bank.to_string())
    .fetch_all(&self.pool)
    .await
    .map_err(|error| map_read("vector index", error))?;

    let mut results = Vec::with_capacity(rows.len());
    for (id, distance) in rows {
      let id = FactId::parse(&id)
        .map_err(|error| Error::fatal(format!("corrupt fact id `{id}`: {error}")))?;
      // vec0's cosine distance is `1 - cosine`; convert back to similarity.
      let score = (1.0 - distance) as f32;
      results.push((id, score.clamp(-1.0, 1.0)));
    }

    results.sort_by(|left, right| {
      right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
    });
    results.truncate(k);

    Ok(results)
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

  async fn facts_in_range(&self, bank: BankId, range: TimeRange) -> Result<Vec<FactId>> {
    // Closed operation interval; an open end (`occurred_end IS NULL`) means
    // "still ongoing", so it intersects every range starting before its start.
    // Rows with no start carry no usable interval.
    let rows: Vec<String> = sqlx::query_scalar(
      "SELECT id FROM fact \
       WHERE bank_id = ? AND occurred_start IS NOT NULL \
         AND (occurred_end IS NULL OR occurred_end >= occurred_start) \
         AND occurred_start <= ? \
         AND (occurred_end IS NULL OR occurred_end >= ?) \
       ORDER BY occurred_start, id",
    )
    .bind(bank.to_string())
    .bind(to_epoch_millis(range.end))
    .bind(to_epoch_millis(range.start))
    .fetch_all(&self.pool)
    .await
    .map_err(|error| map_read("fact", error))?;

    rows
      .into_iter()
      .map(|id| {
        FactId::parse(&id).map_err(|error| Error::fatal(format!("corrupt fact id `{id}`: {error}")))
      })
      .collect()
  }

  async fn fetch_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>> {
    if ids.is_empty() {
      return Ok(Vec::new());
    }

    let mut query = QueryBuilder::<Sqlite>::new(
      "SELECT f.id, f.bank_id, f.fact_type, f.text, v.embedding, f.chunk_id, f.occurred_start, \
              f.occurred_end, f.mentioned_at, f.confidence, f.access_count, f.attrs, \
              f.created_at, f.updated_at \
       FROM fact f LEFT JOIN fact_vec v ON v.rowid = f.rowid WHERE f.id IN (",
    );
    let mut separated = query.separated(", ");
    for id in ids {
      separated.push_bind(id.to_string());
    }
    query.push(")");

    let rows: Vec<FactRow> = query
      .build_query_as()
      .fetch_all(&self.pool)
      .await
      .map_err(|error| map_read("fact", error))?;

    let mut facts = Vec::with_capacity(rows.len());
    for row in rows {
      if let Some(fact) = fact_from_row(row)? {
        facts.push(fact);
      }
    }

    Ok(facts)
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

  async fn source_progress(
    &self,
    bank: BankId,
    source: SourceId,
  ) -> Result<Option<SourceProgress>> {
    // One statement: SQLite gives statement-level snapshot consistency, so the
    // source status, chunk counts, and fact count cannot straddle a concurrent
    // claim/commit.
    let row: Option<(String, i64, i64, i64, i64, i64, Option<i64>, i64)> = sqlx::query_as(
      "WITH c AS ( \
         SELECT \
           COUNT(*) AS total, \
           COUNT(*) FILTER (WHERE status = 'pending') AS pending, \
           COUNT(*) FILTER (WHERE status = 'extracting') AS extracting, \
           COUNT(*) FILTER (WHERE status = 'done') AS done, \
           COUNT(*) FILTER (WHERE status = 'failed') AS failed, \
           MIN(CASE WHEN status = 'extracting' THEN updated_at END) AS oldest_extracting \
         FROM chunk WHERE bank_id = ? AND source_id = ? \
       ) \
       SELECT s.status, c.total, c.pending, c.extracting, c.done, c.failed, \
              c.oldest_extracting, \
              (SELECT COUNT(*) FROM fact f JOIN chunk ch ON ch.id = f.chunk_id \
               WHERE ch.bank_id = ? AND ch.source_id = ?) AS facts_written \
       FROM source s CROSS JOIN c \
       WHERE s.bank_id = ? AND s.id = ?",
    )
    .bind(bank.to_string())
    .bind(source.to_string())
    .bind(bank.to_string())
    .bind(source.to_string())
    .bind(bank.to_string())
    .bind(source.to_string())
    .fetch_optional(&self.pool)
    .await
    .map_err(|error| map_read("source", error))?;

    let Some((status, total, pending, extracting, done, failed, oldest_extracting, facts)) = row
    else {
      return Ok(None);
    };

    let source_status = status
      .parse::<SourceStatus>()
      .map_err(|error| Error::fatal(error.to_string()))?;
    let to_usize = |value: i64, field: &'static str| {
      usize::try_from(value).map_err(|_| Error::fatal(format!("corrupt {field} `{value}`")))
    };

    Ok(Some(SourceProgress {
      source_status,
      chunks_total: to_usize(total, "chunk total")?,
      chunks_pending: to_usize(pending, "pending chunks")?,
      chunks_extracting: to_usize(extracting, "extracting chunks")?,
      chunks_done: to_usize(done, "done chunks")?,
      chunks_failed: to_usize(failed, "failed chunks")?,
      oldest_extracting_at: oldest_extracting.map(timestamp).transpose()?,
      facts_written: to_usize(facts, "fact count")?,
    }))
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

/// Executes tx2: writes every product and flips the chunk status under the
/// claim token.
async fn commit_chunk_tx(tx: &mut Transaction<'_, Sqlite>, commit: &ChunkCommit) -> Result<()> {
  // The DTO carries no bank; the chunk row is the authority.
  let bank: Option<String> = sqlx::query_scalar("SELECT bank_id FROM chunk WHERE id = ?")
    .bind(commit.chunk_id.to_string())
    .fetch_optional(&mut **tx)
    .await
    .map_err(|error| map_read("chunk", error))?;
  let bank = bank.ok_or_else(|| Error::not_found("chunk", commit.chunk_id.to_string()))?;
  let bank = BankId::parse(&bank)
    .map_err(|error| Error::fatal(format!("corrupt bank id `{bank}`: {error}")))?;

  // 1. Entities. A canonical-name conflict reuses the existing row and rewrites
  // the ids referenced by this commit (bindings + statistics).
  let mut remap: HashMap<EntityId, EntityId> = HashMap::new();
  for entity in &commit.entities {
    if entity.bank_id != bank {
      return Err(Error::config(format!(
        "entity {} belongs to bank {}, not {}",
        entity.id, entity.bank_id, bank
      )));
    }

    let inserted = sqlx::query(
      "INSERT INTO entity (id, bank_id, canonical_name, entity_type, mention_count, first_seen, \
       last_seen, attrs, created_at, updated_at) \
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
       ON CONFLICT(bank_id, canonical_name) DO NOTHING",
    )
    .bind(entity.id.to_string())
    .bind(bank.to_string())
    .bind(&entity.canonical_name)
    .bind(entity.entity_type.as_str())
    .bind(i64::from(entity.mention_count))
    .bind(entity.first_seen.map(to_epoch_millis))
    .bind(entity.last_seen.map(to_epoch_millis))
    .bind(entity.attrs.as_ref().map(serde_json::Value::to_string))
    .bind(to_epoch_millis(entity.created_at))
    .bind(to_epoch_millis(entity.updated_at))
    .execute(&mut **tx)
    .await
    .map_err(|error| map_write("entity", error))?;

    if inserted.rows_affected() == 0 {
      let existing: String =
        sqlx::query_scalar("SELECT id FROM entity WHERE bank_id = ? AND canonical_name = ?")
          .bind(bank.to_string())
          .bind(&entity.canonical_name)
          .fetch_one(&mut **tx)
          .await
          .map_err(|error| map_read("entity", error))?;
      let existing = EntityId::parse(&existing)
        .map_err(|error| Error::fatal(format!("corrupt entity id `{existing}`: {error}")))?;
      if existing != entity.id {
        remap.insert(entity.id, existing);
      }
    }
  }

  // 2. Facts, plus the 1:1 vector row when an embedding is available.
  for fact in &commit.facts {
    if fact.bank_id != bank {
      return Err(Error::config(format!(
        "fact {} belongs to bank {}, not {}",
        fact.id, fact.bank_id, bank
      )));
    }

    sqlx::query(
      "INSERT INTO fact (id, bank_id, fact_type, text, chunk_id, occurred_start, occurred_end, \
       mentioned_at, confidence, access_count, attrs, created_at, updated_at) \
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
    )
    .bind(fact.id.to_string())
    .bind(bank.to_string())
    .bind(fact.fact_type.as_str())
    .bind(&fact.text)
    .bind(fact.chunk_id.map(|id| id.to_string()))
    .bind(fact.occurred.start().map(to_epoch_millis))
    .bind(fact.occurred.end().map(to_epoch_millis))
    .bind(fact.mentioned_at.map(to_epoch_millis))
    .bind(fact.confidence)
    .bind(fact.attrs.as_ref().map(serde_json::Value::to_string))
    .bind(to_epoch_millis(fact.created_at))
    .bind(to_epoch_millis(fact.created_at))
    .execute(&mut **tx)
    .await
    .map_err(|error| map_write("fact", error))?;

    if let Some(embedding) = &fact.embedding {
      sqlx::query(
        "INSERT INTO fact_vec(rowid, fact_id, bank_id, embedding) \
         SELECT rowid, id, bank_id, ? FROM fact WHERE id = ?",
      )
      .bind(encode_embedding(embedding))
      .bind(fact.id.to_string())
      .execute(&mut **tx)
      .await
      .map_err(|error| map_write("fact_vec", error))?;
    }
  }

  // 3. Bindings (entity ids rewritten after canonical-name conflicts).
  for binding in &commit.bindings {
    if binding.bank_id != bank {
      return Err(Error::config(format!(
        "binding references bank {}, not {}",
        binding.bank_id, bank
      )));
    }

    let entity_id = remap
      .get(&binding.entity_id)
      .copied()
      .unwrap_or(binding.entity_id);
    sqlx::query(
      "INSERT INTO fact_entity (bank_id, fact_id, entity_id, mention_text, created_at, updated_at) \
       VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(bank.to_string())
    .bind(binding.fact_id.to_string())
    .bind(entity_id.to_string())
    .bind(&binding.mention_text)
    .bind(to_epoch_millis(binding.created_at))
    .bind(to_epoch_millis(binding.created_at))
    .execute(&mut **tx)
    .await
    .map_err(|error| map_write("fact_entity", error))?;
  }

  // 4. Graph edges (temporal pairs were already expanded by the builders).
  for link in &commit.links {
    if link.bank_id != bank {
      return Err(Error::config(format!(
        "link {} belongs to bank {}, not {}",
        link.id, link.bank_id, bank
      )));
    }

    sqlx::query(
      "INSERT INTO fact_link (id, bank_id, from_fact_id, to_fact_id, link_type, relation, weight, \
       created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(link.id.to_string())
    .bind(bank.to_string())
    .bind(link.from.to_string())
    .bind(link.to.to_string())
    .bind(link.link_type.as_str())
    .bind(link.relation.map(|relation| relation.as_str()))
    .bind(link.weight)
    .bind(to_epoch_millis(link.created_at))
    .bind(to_epoch_millis(link.created_at))
    .execute(&mut **tx)
    .await
    .map_err(|error| map_write("fact_link", error))?;
  }

  // 5. Full-text corpus (maintained manually; one row per fact).
  for fact in &commit.facts {
    sqlx::query("INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)")
      .bind(fact.id.to_string())
      .bind(&fact.text)
      .execute(&mut **tx)
      .await
      .map_err(|error| map_write("fact_fts", error))?;
  }

  // 6. Entity statistics deltas, with remapped ids.
  let mut stats = commit.entity_stats.clone();
  for stat in &mut stats {
    if let Some(mapped) = remap.get(&stat.entity_id) {
      stat.entity_id = *mapped;
    }
  }
  apply_entity_stats(tx, bank, &stats).await?;

  // 7. Opinion confidence/text updates.
  for update in &commit.opinion_updates {
    let updated = sqlx::query(
      "UPDATE fact SET confidence = ?, text = COALESCE(?, text), updated_at = ? \
       WHERE id = ? AND bank_id = ? AND fact_type = 'opinion'",
    )
    .bind(update.new_confidence)
    .bind(&update.new_text)
    .bind(to_epoch_millis(update.updated_at))
    .bind(update.opinion_id.to_string())
    .bind(bank.to_string())
    .execute(&mut **tx)
    .await
    .map_err(|error| map_write("fact", error))?;

    if updated.rows_affected() == 0 {
      return Err(Error::fatal(format!(
        "opinion update references missing opinion {}",
        update.opinion_id
      )));
    }
  }

  // 8. Finish the claim: conditional on the token written by claim_chunk.
  let finished = sqlx::query(
    "UPDATE chunk SET status = 'done', updated_at = ? \
     WHERE id = ? AND status = 'extracting' AND updated_at = ?",
  )
  .bind(to_epoch_millis(commit.completed_at))
  .bind(commit.chunk_id.to_string())
  .bind(to_epoch_millis(commit.claimed_at))
  .execute(&mut **tx)
  .await
  .map_err(|error| map_write("chunk", error))?;

  if finished.rows_affected() == 0 {
    return Err(Error::conflict(
      "chunk",
      format!("claim for {} is no longer valid", commit.chunk_id),
    ));
  }

  Ok(())
}

/// Applies entity evidence deltas inside the caller's tx2 transaction.
///
/// `mention_count` counts world/experience bindings only, so the caller supplies
/// one update per binding insert or delete. Deletions decrement the counter but
/// never rewind `first_seen`/`last_seen`: the bounds mean "first/last ever
/// seen", so a re-extraction replacement cannot erase history. A zero-row update
/// means the commit references an entity outside the chunk's bank; that is
/// corrupt input, not a retryable condition.
//
// Called by `commit_chunk` after the binding rows are written.
async fn apply_entity_stats(
  tx: &mut Transaction<'_, Sqlite>,
  bank: BankId,
  updates: &[EntityStatUpdate],
) -> Result<()> {
  for update in updates {
    let delta = if update.add { 1_i64 } else { -1_i64 };
    let updated_at = to_epoch_millis(update.updated_at);

    // Deletions never touch the seen bounds, even when they carry a time: the
    // bounds describe history, and a delete must not rewind them.
    let affected = match (update.add, update.at) {
      (true, Some(at)) => {
        let at = to_epoch_millis(at);
        sqlx::query(
          "UPDATE entity \
           SET mention_count = MAX(mention_count + ?, 0), \
               first_seen = MIN(COALESCE(first_seen, ?), ?), \
               last_seen = MAX(COALESCE(last_seen, ?), ?), \
               updated_at = ? \
           WHERE id = ? AND bank_id = ?",
        )
        .bind(delta)
        .bind(at)
        .bind(at)
        .bind(at)
        .bind(at)
        .bind(updated_at)
        .bind(update.entity_id.to_string())
        .bind(bank.to_string())
        .execute(&mut **tx)
        .await
        .map_err(|error| map_write("entity", error))?
      }
      _ => sqlx::query(
        "UPDATE entity \
           SET mention_count = MAX(mention_count + ?, 0), updated_at = ? \
           WHERE id = ? AND bank_id = ?",
      )
      .bind(delta)
      .bind(updated_at)
      .bind(update.entity_id.to_string())
      .bind(bank.to_string())
      .execute(&mut **tx)
      .await
      .map_err(|error| map_write("entity", error))?,
    };

    if affected.rows_affected() == 0 {
      return Err(Error::fatal(format!(
        "entity stat update references entity {} outside bank {}",
        update.entity_id, bank
      )));
    }
  }

  Ok(())
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

type FactRow = (
  String,
  String,
  String,
  String,
  Option<Vec<u8>>,
  Option<String>,
  Option<i64>,
  Option<i64>,
  Option<i64>,
  Option<f64>,
  i64,
  Option<String>,
  i64,
  i64,
);

fn fact_from_row(row: FactRow) -> Result<Option<Fact>> {
  let (
    id,
    bank_id,
    fact_type,
    text,
    embedding,
    chunk_id,
    occurred_start,
    occurred_end,
    mentioned_at,
    confidence,
    access_count,
    attrs,
    created_at,
    updated_at,
  ) = row;

  let id =
    FactId::parse(&id).map_err(|error| Error::fatal(format!("corrupt fact id `{id}`: {error}")))?;

  // Invalid occurrence bounds (legacy or externally written rows) make this
  // single fact unusable, but must not fail the whole read: skip it with a
  // warning and keep the other rows.
  let occurred = match OccurrenceInterval::from_parts(
    occurred_start.map(timestamp).transpose()?,
    occurred_end.map(timestamp).transpose()?,
  ) {
    Ok(occurred) => occurred,
    Err(error) => {
      warn!(
        fact = %id,
        %error,
        "skipping fact row with an invalid occurrence interval"
      );
      return Ok(None);
    }
  };

  Ok(Some(Fact {
    id,
    bank_id: BankId::parse(&bank_id)
      .map_err(|error| Error::fatal(format!("corrupt bank id `{bank_id}`: {error}")))?,
    fact_type: fact_type
      .parse::<FactType>()
      .map_err(|error| Error::fatal(error.to_string()))?,
    text,
    embedding: decode_embedding(embedding)?,
    chunk_id: chunk_id
      .map(|value| {
        ChunkId::parse(&value)
          .map_err(|error| Error::fatal(format!("corrupt chunk id `{value}`: {error}")))
      })
      .transpose()?,
    occurred,
    mentioned_at: mentioned_at.map(timestamp).transpose()?,
    confidence: confidence.map(|value| value as f32),
    access_count: u32::try_from(access_count)
      .map_err(|_| Error::fatal(format!("corrupt access count `{access_count}`")))?,
    attrs: attrs
      .map(|text| {
        serde_json::from_str(&text)
          .map_err(|error| Error::fatal(format!("corrupt fact attrs: {error}")))
      })
      .transpose()?,
    created_at: timestamp(created_at)?,
    updated_at: timestamp(updated_at)?,
  }))
}

/// Decodes a little-endian `f32` BLOB.
///
/// The exact BLOB layout is adapter-owned (the logical schema only requires a
/// fixed-size vector); the tx2 write path must use this same encoding.
fn decode_embedding(bytes: Option<Vec<u8>>) -> Result<Option<Vec<f32>>> {
  bytes
    .map(|bytes| decode_embedding_bytes(&bytes))
    .transpose()
}

fn decode_embedding_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
  if !bytes.len().is_multiple_of(4) {
    return Err(Error::fatal(format!(
      "corrupt embedding blob: {} bytes is not a multiple of 4",
      bytes.len()
    )));
  }

  Ok(
    bytes
      .chunks_exact(4)
      .map(|chunk| {
        let mut word = [0_u8; 4];
        word.copy_from_slice(chunk);
        f32::from_le_bytes(word)
      })
      .collect(),
  )
}

/// Encodes a vector in sqlite-vec's compact float32 layout (little-endian).
fn encode_embedding(values: &[f32]) -> Vec<u8> {
  values
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect()
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::memory_pool;

  async fn create_entity(store: &SqliteStore, bank: BankId, name: &str) -> EntityId {
    store
      .ensure_entity(NewEntity {
        id: EntityId::new(),
        bank_id: bank,
        canonical_name: name.into(),
        entity_type: EntityKind::Person,
        created_at: timestamp(0).expect("created_at"),
      })
      .await
      .expect("create entity")
      .0
  }

  fn update(entity_id: EntityId, add: bool, at: Option<i64>, updated_at: i64) -> EntityStatUpdate {
    EntityStatUpdate {
      entity_id,
      add,
      at: at.map(|millis| timestamp(millis).expect("at")),
      updated_at: timestamp(updated_at).expect("updated_at"),
    }
  }

  async fn apply(pool: &SqlitePool, bank: BankId, updates: &[EntityStatUpdate]) -> Result<()> {
    let mut tx = pool.begin().await.expect("begin");
    if let Err(error) = apply_entity_stats(&mut tx, bank, updates).await {
      tx.rollback().await.expect("rollback");
      return Err(error);
    }
    tx.commit().await.expect("commit");
    Ok(())
  }

  async fn read_stats(pool: &SqlitePool, id: EntityId) -> (i64, Option<i64>, Option<i64>) {
    sqlx::query_as("SELECT mention_count, first_seen, last_seen FROM entity WHERE id = ?")
      .bind(id.to_string())
      .fetch_one(pool)
      .await
      .expect("read stats")
  }

  #[tokio::test]
  async fn add_tracks_count_and_seen_bounds() {
    let pool = memory_pool().await.expect("pool");
    let store = SqliteStore::new(pool.clone());
    let bank = BankId::new();
    let id = create_entity(&store, bank, "emily").await;

    apply(&pool, bank, &[update(id, true, Some(1_000), 1_100)])
      .await
      .expect("add");
    assert_eq!(read_stats(&pool, id).await, (1, Some(1_000), Some(1_000)));

    apply(&pool, bank, &[update(id, true, Some(500), 1_200)])
      .await
      .expect("earlier add");
    assert_eq!(read_stats(&pool, id).await, (2, Some(500), Some(1_000)));

    apply(&pool, bank, &[update(id, true, Some(2_000), 1_300)])
      .await
      .expect("later add");
    assert_eq!(read_stats(&pool, id).await, (3, Some(500), Some(2_000)));
  }

  #[tokio::test]
  async fn remove_decrements_without_rewinding_bounds() {
    let pool = memory_pool().await.expect("pool");
    let store = SqliteStore::new(pool.clone());
    let bank = BankId::new();
    let id = create_entity(&store, bank, "emily").await;

    apply(
      &pool,
      bank,
      &[
        update(id, true, Some(1_000), 1_000),
        update(id, true, Some(2_000), 2_000),
      ],
    )
    .await
    .expect("add twice");
    assert_eq!(read_stats(&pool, id).await, (2, Some(1_000), Some(2_000)));

    apply(&pool, bank, &[update(id, false, Some(500), 3_000)])
      .await
      .expect("remove");
    assert_eq!(
      read_stats(&pool, id).await,
      (1, Some(1_000), Some(2_000)),
      "removal does not rewind the seen bounds, even with an earlier time"
    );

    apply(
      &pool,
      bank,
      &[
        update(id, false, None, 4_000),
        update(id, false, None, 5_000),
      ],
    )
    .await
    .expect("remove below zero");
    assert_eq!(
      read_stats(&pool, id).await,
      (0, Some(1_000), Some(2_000)),
      "mention_count is floored at zero"
    );
  }

  #[tokio::test]
  async fn null_time_leaves_seen_bounds() {
    let pool = memory_pool().await.expect("pool");
    let store = SqliteStore::new(pool.clone());
    let bank = BankId::new();
    let id = create_entity(&store, bank, "emily").await;

    apply(&pool, bank, &[update(id, true, None, 1_000)])
      .await
      .expect("add without time");
    assert_eq!(read_stats(&pool, id).await, (1, None, None));

    apply(&pool, bank, &[update(id, true, Some(1_000), 2_000)])
      .await
      .expect("add with time");
    assert_eq!(read_stats(&pool, id).await, (2, Some(1_000), Some(1_000)));

    apply(&pool, bank, &[update(id, false, None, 3_000)])
      .await
      .expect("remove without time");
    assert_eq!(read_stats(&pool, id).await, (1, Some(1_000), Some(1_000)));
  }

  #[tokio::test]
  async fn unknown_entity_is_fatal_and_rolls_back() {
    let pool = memory_pool().await.expect("pool");
    let store = SqliteStore::new(pool.clone());
    let bank = BankId::new();
    let id = create_entity(&store, bank, "emily").await;

    let error = apply(
      &pool,
      bank,
      &[
        update(id, true, Some(1_000), 1_000),
        update(EntityId::new(), true, Some(1_000), 1_000),
      ],
    )
    .await
    .expect_err("unknown entity");

    assert!(matches!(error, Error::Fatal(_)), "error: {error}");
    assert_eq!(
      read_stats(&pool, id).await,
      (0, None, None),
      "the caller's transaction rolled back the earlier update"
    );
  }

  #[tokio::test]
  async fn entity_from_another_bank_is_fatal() {
    let pool = memory_pool().await.expect("pool");
    let store = SqliteStore::new(pool.clone());
    let bank = BankId::new();
    let id = create_entity(&store, bank, "emily").await;

    let error = apply(
      &pool,
      BankId::new(),
      &[update(id, true, Some(1_000), 1_000)],
    )
    .await
    .expect_err("wrong bank");

    assert!(matches!(error, Error::Fatal(_)), "error: {error}");
    assert_eq!(read_stats(&pool, id).await, (0, None, None));
  }

  #[tokio::test]
  async fn count_stays_consistent_with_evidence_bindings() {
    let pool = memory_pool().await.expect("pool");
    let store = SqliteStore::new(pool.clone());
    let bank = BankId::new();
    let id = create_entity(&store, bank, "emily").await;

    let mut tx = pool.begin().await.expect("begin");
    let mut updates = Vec::new();
    for _ in 0..3 {
      let fact = FactId::new();
      sqlx::query(
        "INSERT INTO fact (id, bank_id, fact_type, text, created_at, updated_at) \
         VALUES (?, ?, 'world', 'fact', 0, 0)",
      )
      .bind(fact.to_string())
      .bind(bank.to_string())
      .execute(&mut *tx)
      .await
      .expect("insert fact");
      sqlx::query(
        "INSERT INTO fact_entity (bank_id, fact_id, entity_id, mention_text, created_at, updated_at) \
         VALUES (?, ?, ?, 'Emily', 0, 0)",
      )
      .bind(bank.to_string())
      .bind(fact.to_string())
      .bind(id.to_string())
      .execute(&mut *tx)
      .await
      .expect("insert binding");
      updates.push(update(id, true, Some(1_000), 1_000));
    }
    apply_entity_stats(&mut tx, bank, &updates)
      .await
      .expect("apply stats");
    tx.commit().await.expect("commit");

    let bindings: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM fact_entity WHERE entity_id = ?")
      .bind(id.to_string())
      .fetch_one(&pool)
      .await
      .expect("count bindings");
    assert_eq!(read_stats(&pool, id).await.0, bindings);
  }
}
