//! Retain entry point and per-source progress.

use std::sync::Arc;
use std::time::Duration;

use engine::ports::{Clock, NewChunk, SourceProgress, Store, now_ts};
use engine::retain::{ChunkConfig, RawPayload, chunk, normalize};
use engine::tokens::CharHeuristic;
use engine::types::{BankId, ChunkId, Source, SourceId, SourceStatus};
use engine::{Error, Result};

/// Retain settings. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct RetainConfig {
  /// Chunking parameters.
  pub chunk: ChunkConfig,
  /// An `extracting` chunk older than this marks the source as `Stalled`.
  pub stale_after: Duration,
}

impl Default for RetainConfig {
  fn default() -> Self {
    Self {
      chunk: ChunkConfig::default(),
      stale_after: Duration::from_secs(10 * 60),
    }
  }
}

/// Where a source is in the async retain contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainState {
  /// Persisted but chunking has not completed (crash between tx0 and tx1).
  Received,
  /// Chunks are queued; no extraction is running yet.
  Queued,
  /// At least one chunk is currently claimed.
  Running,
  /// An `extracting` chunk exceeded `stale_after`.
  Stalled,
  /// Every chunk completed successfully (zero-chunk sources count as complete).
  Completed,
  /// Some chunks completed and some were dead-lettered.
  CompletedWithFailures,
  /// Every chunk was dead-lettered.
  Failed,
}

/// Per-source retain progress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainStatus {
  /// Derived lifecycle state.
  pub state: RetainState,
  /// Total chunk rows.
  pub chunks_total: usize,
  /// Chunks awaiting extraction.
  pub chunks_pending: usize,
  /// Chunks currently claimed.
  pub chunks_extracting: usize,
  /// Chunks successfully committed.
  pub chunks_done: usize,
  /// Chunks dead-lettered.
  pub chunks_failed: usize,
  /// Facts committed from this source's chunks.
  pub facts_written: usize,
}

/// Facade over a [`Store`] and a [`Clock`].
#[derive(Clone)]
pub struct MemoryBank {
  store: Arc<dyn Store>,
  clock: Arc<dyn Clock>,
  config: RetainConfig,
}

impl MemoryBank {
  /// Wires the storage and clock ports with retain configuration.
  pub fn new(store: Arc<dyn Store>, clock: Arc<dyn Clock>, config: RetainConfig) -> Self {
    Self {
      store,
      clock,
      config,
    }
  }

  /// Ingests a payload (async ack).
  ///
  /// Normalizes, writes the source (tx0), chunks, then writes the chunks and
  /// flips the source to `chunked` (tx1). The returned id means "durably
  /// queued", not "extracted": model work starts only in the worker. Errors
  /// before the ack are returned directly; later failures surface through
  /// [`MemoryBank::retain_status`].
  pub async fn retain(&self, bank: BankId, payload: RawPayload) -> Result<SourceId> {
    let title = payload.title().map(str::to_owned);
    let normalized = normalize(payload);
    let now = now_ts(self.clock.as_ref());
    let source_id = SourceId::new();

    self
      .store
      .create_source(Source {
        id: source_id,
        bank_id: bank,
        title,
        payload_text: normalized.text.clone(),
        status: SourceStatus::Received,
        created_at: now,
        updated_at: now,
      })
      .await?;

    let chunks = chunk(&normalized, &self.config.chunk, &CharHeuristic::default())
      .into_iter()
      .map(|raw| NewChunk {
        id: ChunkId::new(),
        bank_id: bank,
        source_id,
        chunk_index: raw.index,
        text: raw.text,
        content_hash: raw.content_hash,
        time_start: raw.time_start,
        time_end: raw.time_end,
        created_at: now,
      })
      .collect();

    // An empty batch still flips the source to `chunked` (all hashes collided
    // or the payload produced no chunks).
    self.store.append_chunks(source_id, chunks).await?;

    Ok(source_id)
  }

  /// Returns the derived progress of one source.
  pub async fn retain_status(&self, bank: BankId, source: SourceId) -> Result<RetainStatus> {
    let progress = self
      .store
      .source_progress(bank, source)
      .await?
      .ok_or_else(|| Error::not_found("source", source.to_string()))?;

    let now = now_ts(self.clock.as_ref());
    let stale_ms = i64::try_from(self.config.stale_after.as_millis()).unwrap_or(i64::MAX);
    let stalled = progress.oldest_extracting_at.is_some_and(|oldest| {
      now
        .timestamp_millis()
        .saturating_sub(oldest.timestamp_millis())
        > stale_ms
    });

    let state = derive_state(&progress, stalled);

    Ok(RetainStatus {
      state,
      chunks_total: progress.chunks_total,
      chunks_pending: progress.chunks_pending,
      chunks_extracting: progress.chunks_extracting,
      chunks_done: progress.chunks_done,
      chunks_failed: progress.chunks_failed,
      facts_written: progress.facts_written,
    })
  }
}

fn derive_state(progress: &SourceProgress, stalled: bool) -> RetainState {
  if progress.chunks_total == 0 {
    return match progress.source_status {
      SourceStatus::Received => RetainState::Received,
      SourceStatus::Chunked => RetainState::Completed,
    };
  }

  if stalled {
    RetainState::Stalled
  } else if progress.chunks_extracting > 0 {
    RetainState::Running
  } else if progress.chunks_pending > 0 {
    RetainState::Queued
  } else if progress.chunks_failed == 0 {
    RetainState::Completed
  } else if progress.chunks_done == 0 {
    RetainState::Failed
  } else {
    RetainState::CompletedWithFailures
  }
}
