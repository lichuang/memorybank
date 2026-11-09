//! Background extraction worker and deterministic rescan pass.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use engine::link::{LinkConfig, build_semantic_links, build_temporal_links, causal_links};
use engine::ports::{ChunkCommit, EntityStatUpdate, NewBinding, NewChunk, NewFact, now_ts};
use engine::resolve::{MentionContext, ResolveWeights, resolve};
use engine::retain::{
  ExtractionConfig, NormalizedPayload, chunk, embed_facts, extract_facts, fact_attrs, fact_text,
};
use engine::tokens::CharHeuristic;
use engine::types::{BankId, Chunk, ChunkId, EntityId, Fact, FactId, OccurrenceInterval};
use engine::{Error, Result};
use tokio::task::JoinHandle;
use tracing::warn;

use crate::retain::MemoryBank;

/// Pipeline configuration for the extraction worker.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineConfig {
  /// Extraction call parameters.
  pub extraction: ExtractionConfig,
  /// Link thresholds and budgets.
  pub links: LinkConfig,
  /// Entity resolution weights.
  pub resolve: ResolveWeights,
  /// Embedding batch size.
  pub embedding_batch: usize,
}

impl Default for PipelineConfig {
  fn default() -> Self {
    Self {
      extraction: ExtractionConfig::default(),
      links: LinkConfig::default(),
      resolve: ResolveWeights::default(),
      embedding_batch: 32,
    }
  }
}

/// Worker loop settings. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerConfig {
  /// Reserved for parallel chunk processing; the current pass is sequential.
  pub concurrency: usize,
  /// Periodic sweep interval.
  pub poll_interval: Duration,
  /// Claim lease after which an `extracting` chunk is reclaimable.
  pub stale_after: Duration,
  /// Attempts before a failing chunk is dead-lettered.
  pub max_attempts: u32,
  /// Backoff used by the spawned loop after a pass with failures.
  pub retry_delay: Duration,
}

impl Default for WorkerConfig {
  fn default() -> Self {
    Self {
      concurrency: 1,
      poll_interval: Duration::from_secs(5),
      stale_after: Duration::from_secs(10 * 60),
      max_attempts: 3,
      retry_delay: Duration::from_secs(1),
    }
  }
}

/// Outcome of one deterministic rescan pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RescanReport {
  /// `received` sources that were re-chunked.
  pub sources_chunked: usize,
  /// Chunks committed successfully.
  pub chunks_processed: usize,
  /// Chunks dead-lettered after exhausting attempts.
  pub failed: usize,
}

impl MemoryBank {
  /// Runs one deterministic pass over `bank` using the stored worker config.
  pub async fn rescan(&self, bank: BankId) -> Result<RescanReport> {
    let config = self.worker.clone();
    self.rescan_with(bank, &config).await
  }

  /// Spawns the periodic worker. It scans every bank, sleeps for
  /// `poll_interval`, and wakes early on [`MemoryBank::retain`].
  pub fn spawn_worker(&self, config: WorkerConfig) -> JoinHandle<()> {
    let this = self.clone();
    tokio::spawn(async move {
      loop {
        let mut had_failure = false;

        match this.store.list_banks().await {
          Ok(banks) => {
            for bank in banks {
              match this.rescan_with(bank, &config).await {
                Ok(report) => had_failure |= report.failed > 0,
                Err(error) => {
                  had_failure = true;
                  warn!(%bank, %error, "worker rescan failed");
                }
              }
            }
          }
          Err(error) => {
            had_failure = true;
            warn!(%error, "worker could not list banks");
          }
        }

        let delay = if had_failure {
          config.retry_delay
        } else {
          config.poll_interval
        };
        tokio::select! {
          _ = tokio::time::sleep(delay) => {}
          _ = this.notify.notified() => {}
        }
      }
    })
  }

  async fn rescan_with(&self, bank: BankId, config: &WorkerConfig) -> Result<RescanReport> {
    let mut sources_chunked = 0;

    let work = self
      .store
      .pending_work(bank, now_ts(self.clock.as_ref()), config.stale_after)
      .await?;

    // A crash between tx0 and tx1 leaves `received`; re-chunk from the payload
    // (turn spans are gone, so this uses the window fallback).
    for source_id in work.sources_received {
      let source = self.store.load_source(bank, source_id).await?;
      let normalized = NormalizedPayload {
        text: source.payload_text.clone(),
        spans: Vec::new(),
      };
      let now = now_ts(self.clock.as_ref());
      let chunks: Vec<NewChunk> = chunk(&normalized, &self.config.chunk, &CharHeuristic::default())
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

      self.store.append_chunks(source_id, chunks).await?;
      sources_chunked += 1;
    }

    let mut attempts: HashMap<ChunkId, u32> = HashMap::new();
    let mut chunks_processed = 0;
    let mut failed = 0;

    loop {
      let now = now_ts(self.clock.as_ref());
      let Some(claimed) = self
        .store
        .claim_chunk(bank, now, config.stale_after)
        .await?
      else {
        break;
      };
      let chunk = claimed.chunk;
      let attempt = attempts.entry(chunk.id).or_insert(0);
      *attempt += 1;

      match self.process_chunk(bank, &chunk).await {
        Ok(()) => chunks_processed += 1,
        Err(error) => {
          if matches!(error, Error::Conflict(_, _)) {
            warn!(chunk = %chunk.id, %error, "claim was stolen; leaving the chunk alone");
            continue;
          }

          warn!(
            chunk = %chunk.id,
            attempt = *attempt,
            %error,
            "chunk processing failed"
          );

          if *attempt >= config.max_attempts.max(1) {
            if let Err(fail_error) = self.store.fail_chunk(chunk.id, chunk.updated_at, now).await {
              warn!(chunk = %chunk.id, %fail_error, "could not dead-letter chunk");
            }
            failed += 1;
          } else {
            self
              .store
              .release_chunk(chunk.id, chunk.updated_at, now)
              .await?;
          }
        }
      }
    }

    Ok(RescanReport {
      sources_chunked,
      chunks_processed,
      failed,
    })
  }

  async fn process_chunk(&self, bank: BankId, chunk: &Chunk) -> Result<()> {
    let llm = self
      .llm
      .as_ref()
      .ok_or_else(|| Error::config("worker requires an Llm"))?;
    let embedder = self
      .embedder
      .as_ref()
      .ok_or_else(|| Error::config("worker requires an Embedder"))?;

    let report = extract_facts(llm.as_ref(), chunk, &self.pipeline.extraction).await?;
    let now = now_ts(self.clock.as_ref());

    let mut facts = Vec::with_capacity(report.facts.len());
    for extracted in &report.facts {
      let occurred =
        OccurrenceInterval::from_parts(extracted.occurred_start, extracted.occurred_end)
          .map_err(|error| Error::fatal(format!("invalid extracted interval: {error}")))?;
      facts.push(Fact {
        id: FactId::new(),
        bank_id: bank,
        fact_type: extracted.fact_type,
        text: fact_text(extracted),
        embedding: None,
        chunk_id: Some(chunk.id),
        occurred,
        mentioned_at: extracted.mentioned_at,
        confidence: extracted.confidence,
        access_count: 0,
        attrs: fact_attrs(extracted),
        created_at: now,
        updated_at: now,
      });
    }

    let texts: Vec<String> = facts.iter().map(|fact| fact.text.clone()).collect();
    let embeddings = embed_facts(embedder.as_ref(), &texts, self.pipeline.embedding_batch).await?;
    for (fact, embedding) in facts.iter_mut().zip(embeddings) {
      fact.embedding = embedding.into_option();
    }

    let mut bindings = Vec::new();
    let mut entity_stats = Vec::new();
    let mut entity_ids: HashSet<EntityId> = HashSet::new();
    let mut resolved: Vec<EntityId> = Vec::new();

    for (fact, extracted) in facts.iter().zip(&report.facts) {
      for mention in &extracted.entities {
        let context = MentionContext {
          co_mentioned: resolved.clone(),
          at: fact.mentioned_at,
        };
        let entity_id = resolve(
          self.store.as_ref(),
          bank,
          mention,
          &context,
          &self.pipeline.resolve,
          now,
        )
        .await?;

        resolved.push(entity_id);
        entity_ids.insert(entity_id);
        bindings.push(NewBinding {
          bank_id: bank,
          fact_id: fact.id,
          entity_id,
          mention_text: Some(mention.text.clone()),
          created_at: now,
        });
        entity_stats.push(EntityStatUpdate {
          entity_id,
          add: true,
          at: fact.mentioned_at,
          updated_at: now,
        });
      }
    }

    let mut ids: Vec<EntityId> = entity_ids.into_iter().collect();
    ids.sort_unstable();
    let entities = self.store.fetch_entities(&ids).await?;

    let mut links =
      build_temporal_links(self.store.as_ref(), bank, &facts, &self.pipeline.links, now).await?;
    links.extend(
      build_semantic_links(self.store.as_ref(), bank, &facts, &self.pipeline.links, now).await?,
    );
    let fact_ids: Vec<FactId> = facts.iter().map(|fact| fact.id).collect();
    links.extend(causal_links(bank, &report.facts, &fact_ids, now));

    let new_facts: Vec<NewFact> = facts.iter().map(new_fact).collect();
    self
      .store
      .commit_chunk(ChunkCommit {
        chunk_id: chunk.id,
        claimed_at: chunk.updated_at,
        completed_at: now,
        facts: new_facts,
        entities,
        bindings,
        links,
        entity_stats,
        opinion_updates: Vec::new(),
      })
      .await
  }
}

fn new_fact(fact: &Fact) -> NewFact {
  NewFact {
    id: fact.id,
    bank_id: fact.bank_id,
    fact_type: fact.fact_type,
    text: fact.text.clone(),
    chunk_id: fact.chunk_id,
    embedding: fact.embedding.clone(),
    occurred: fact.occurred,
    mentioned_at: fact.mentioned_at,
    confidence: fact.confidence,
    attrs: fact.attrs.clone(),
    created_at: fact.created_at,
  }
}
