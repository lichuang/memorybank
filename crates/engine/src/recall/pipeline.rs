//! Recall orchestration: run the enabled channels, fuse, rerank, and pack.
//!
//! This module is pure orchestration over injected ports. It embeds the query
//! once, runs the channels that the caller enabled, fuses their ranked lists
//! with RRF, optionally reranks the head, packs the result into the token
//! budget, and reports which channel produced each kept fact.

use std::collections::HashMap;

use tracing::warn;

use crate::ports::{Embedder, FactScope, Reranker, Store, TemporalParser};
use crate::recall::bm25::{Bm25Config, bm25_channel};
use crate::recall::budget::pack;
use crate::recall::graph::{GraphConfig, spread};
use crate::recall::products::{Channel, RecallResult, RecalledFact, sort_channels};
use crate::recall::rerank::rerank_top;
use crate::recall::rrf::{DEFAULT_RRF_K, rrf};
use crate::recall::semantic::{SemanticConfig, semantic_channel};
use crate::recall::temporal::{TemporalConfig, temporal_channel};
use crate::tokens::TokenEstimator;
use crate::types::{BankId, Fact, FactId, TimeRange, Timestamp};
use crate::{Error, Result};

/// Which channels a recall call may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelMask {
  /// Vector channel.
  pub semantic: bool,
  /// Lexical BM25 channel.
  pub bm25: bool,
  /// Graph spreading-activation channel.
  pub graph: bool,
  /// Temporal channel.
  pub temporal: bool,
}

impl ChannelMask {
  /// Every channel enabled.
  pub const ALL: Self = Self {
    semantic: true,
    bm25: true,
    graph: true,
    temporal: true,
  };

  /// No channel enabled.
  pub const NONE: Self = Self {
    semantic: false,
    bm25: false,
    graph: false,
    temporal: false,
  };

  /// Enables exactly one channel.
  pub fn only(channel: Channel) -> Self {
    let mut mask = Self::NONE;
    match channel {
      Channel::Semantic => mask.semantic = true,
      Channel::Bm25 => mask.bm25 = true,
      Channel::Graph => mask.graph = true,
      Channel::Temporal => mask.temporal = true,
    }
    mask
  }

  /// Whether the mask enables `channel`.
  pub fn contains(&self, channel: Channel) -> bool {
    match channel {
      Channel::Semantic => self.semantic,
      Channel::Bm25 => self.bm25,
      Channel::Graph => self.graph,
      Channel::Temporal => self.temporal,
    }
  }

  /// Number of enabled channels.
  pub fn count(&self) -> usize {
    usize::from(self.semantic)
      + usize::from(self.bm25)
      + usize::from(self.graph)
      + usize::from(self.temporal)
  }

  /// Whether no channel is enabled.
  pub fn is_empty(&self) -> bool {
    self.count() == 0
  }
}

/// Per-channel settings and budgets for one recall call.
#[derive(Debug, Clone, PartialEq)]
pub struct RecallOptions {
  /// Hard token cap for the returned evidence (paper eq. 8).
  pub token_budget: usize,
  /// Channels to run.
  pub channels: ChannelMask,
  /// Candidates requested per channel before fusion.
  pub per_channel_k: usize,
  /// Rerank only this prefix; `0` disables reranking.
  pub rerank_top_n: usize,
  /// RRF constant.
  pub rrf_k: u32,
  /// Target networks for retrieval.
  pub scope: FactScope,
  /// Explicit temporal constraint; bypasses the parser when set.
  pub time_range: Option<TimeRange>,
  /// Semantic channel parameters.
  pub semantic: SemanticConfig,
  /// BM25 channel parameters.
  pub bm25: Bm25Config,
  /// Graph channel parameters.
  pub graph: GraphConfig,
  /// Temporal channel parameters.
  pub temporal: TemporalConfig,
}

impl Default for RecallOptions {
  fn default() -> Self {
    Self {
      token_budget: 2_000,
      channels: ChannelMask::ALL,
      per_channel_k: 50,
      rerank_top_n: 20,
      rrf_k: DEFAULT_RRF_K,
      scope: FactScope::All,
      time_range: None,
      semantic: SemanticConfig::default(),
      bm25: Bm25Config::default(),
      graph: GraphConfig::default(),
      temporal: TemporalConfig::default(),
    }
  }
}

/// Runs the recall pipeline for `query`.
///
/// The query is embedded **once**, and each enabled channel is limited to
/// `per_channel_k` candidates. The graph channel is seeded from the semantic
/// hits, so it requires the semantic channel; the temporal channel uses
/// `options.time_range` when set and otherwise asks the injected parser, staying
/// inactive when the query has no time expression. Fused scores are rank-based
/// (RRF), the head may be reranked, and the result is packed into
/// `token_budget`. No channel enabled returns an empty result without touching
/// the store.
///
/// Access counters are the caller's concern: `increment_access` is not part of
/// this pure function.
#[allow(clippy::too_many_arguments)]
pub async fn recall(
  store: &dyn Store,
  embedder: &dyn Embedder,
  reranker: &dyn Reranker,
  temporal_parser: &dyn TemporalParser,
  estimator: &dyn TokenEstimator,
  bank: BankId,
  query: &str,
  now: Timestamp,
  options: &RecallOptions,
) -> Result<RecallResult> {
  if options.channels.is_empty() {
    return Ok(empty_result());
  }

  let embedding = if options.channels.semantic {
    embed_one(embedder, query).await?
  } else {
    None
  };

  let mut lists: Vec<(Channel, Vec<FactId>)> = Vec::new();

  if options.channels.semantic {
    let embedding = embedding.as_ref().ok_or_else(|| {
      Error::config("the semantic channel is enabled but embedding produced no vector")
    })?;
    let hits = semantic_channel(store, bank, embedding, &options.semantic, &options.scope).await?;
    lists.push((Channel::Semantic, take_hits(hits, options.per_channel_k)));
  }

  if options.channels.bm25 {
    let hits = bm25_channel(store, bank, query, &options.bm25, &options.scope).await?;
    lists.push((Channel::Bm25, take_hits(hits, options.per_channel_k)));
  }

  if options.channels.graph {
    let seeds = semantic_seeds(&lists);
    if seeds.is_empty() {
      warn!("graph channel has no semantic seeds; skipping");
    } else {
      let hits = spread(store, bank, &seeds, &options.graph).await?;
      lists.push((Channel::Graph, take_hits(hits, options.per_channel_k)));
    }
  }

  if options.channels.temporal {
    let hits = match options.time_range {
      Some(range) => {
        let parser = FixedRangeParser(range);
        temporal_channel(
          store,
          bank,
          "",
          now,
          &parser,
          &options.temporal,
          &options.scope,
        )
        .await?
      }
      None => {
        temporal_channel(
          store,
          bank,
          query,
          now,
          temporal_parser,
          &options.temporal,
          &options.scope,
        )
        .await?
      }
    };
    lists.push((Channel::Temporal, take_hits(hits, options.per_channel_k)));
  }

  let channel_ranks = collect_ranks(&lists);
  let ranked_lists: Vec<Vec<FactId>> = lists.into_iter().map(|(_, ids)| ids).collect();
  let fused = rrf(&ranked_lists, options.rrf_k);
  if fused.is_empty() {
    return Ok(empty_result());
  }

  let scores: HashMap<FactId, f32> = fused.iter().copied().collect();
  let ids: Vec<FactId> = fused.iter().map(|(id, _)| *id).collect();
  let mut loaded: Vec<Fact> = store.fetch_facts(&ids).await?;
  // `fetch_facts` does not promise an order; restore the fused one.
  let positions: HashMap<FactId, usize> = ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
  loaded.sort_by_key(|fact| positions.get(&fact.id).copied().unwrap_or(usize::MAX));

  let reranked: Vec<Fact> = if options.rerank_top_n == 0 {
    loaded
  } else {
    rerank_top(reranker, query, &loaded, options.rerank_top_n).await
  };

  let ranked: Vec<(Fact, f32)> = reranked
    .into_iter()
    .map(|fact| {
      let score = scores.get(&fact.id).copied().unwrap_or(0.0);
      (fact, score)
    })
    .collect();
  let (packed, tokens_used) = pack(&ranked, options.token_budget, estimator);

  let facts: Vec<RecalledFact> = packed
    .into_iter()
    .map(|fact| {
      let score = scores.get(&fact.id).copied().unwrap_or(0.0);
      let mut channels = channel_ranks.get(&fact.id).cloned().unwrap_or_default();
      sort_channels(&mut channels);
      RecalledFact {
        fact,
        score,
        channels,
      }
    })
    .collect();

  Ok(RecallResult { facts, tokens_used })
}

fn empty_result() -> RecallResult {
  RecallResult {
    facts: Vec::new(),
    tokens_used: 0,
  }
}

/// Truncates a channel's hits to `per_channel_k` and keeps the ids.
fn take_hits(hits: Vec<(FactId, f32)>, per_channel_k: usize) -> Vec<FactId> {
  hits
    .into_iter()
    .take(per_channel_k)
    .map(|(id, _)| id)
    .collect()
}

/// Builds `fact -> [(channel, rank)]` provenance from the per-channel lists.
fn collect_ranks(lists: &[(Channel, Vec<FactId>)]) -> HashMap<FactId, Vec<(Channel, u32)>> {
  let mut ranks: HashMap<FactId, Vec<(Channel, u32)>> = HashMap::new();
  for (channel, ids) in lists {
    for (index, id) in ids.iter().enumerate() {
      let rank = u32::try_from(index + 1).unwrap_or(u32::MAX);
      ranks.entry(*id).or_default().push((*channel, rank));
    }
  }
  ranks
}

/// Seeds for spreading activation: the semantic list, scored by reciprocal rank.
fn semantic_seeds(lists: &[(Channel, Vec<FactId>)]) -> Vec<(FactId, f32)> {
  lists
    .iter()
    .find(|(channel, _)| *channel == Channel::Semantic)
    .map(|(_, ids)| {
      ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, 1.0 / (index + 1) as f32))
        .collect()
    })
    .unwrap_or_default()
}

/// Embeds exactly one query and validates the dimension.
async fn embed_one(embedder: &dyn Embedder, query: &str) -> Result<Option<Vec<f32>>> {
  let mut vectors = embedder.embed(&[query]).await?;
  match vectors.pop() {
    Some(vector) if vector.len() == embedder.dim() => Ok(Some(vector)),
    Some(vector) => Err(Error::config(format!(
      "query embedding has {} dimensions, the embedder declares {}",
      vector.len(),
      embedder.dim()
    ))),
    None => Ok(None),
  }
}

/// A parser that always returns one explicit range (caller-supplied constraint).
struct FixedRangeParser(TimeRange);

impl TemporalParser for FixedRangeParser {
  fn parse(&self, _query: &str, _now: Timestamp) -> Option<TimeRange> {
    Some(self.0)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn channel_mask_helpers() {
    assert_eq!(ChannelMask::ALL.count(), 4);
    assert!(ChannelMask::ALL.contains(Channel::Graph));
    assert!(ChannelMask::NONE.is_empty());
    assert!(ChannelMask::NONE.count() == 0);

    for channel in [
      Channel::Semantic,
      Channel::Bm25,
      Channel::Graph,
      Channel::Temporal,
    ] {
      let mask = ChannelMask::only(channel);
      assert_eq!(mask.count(), 1);
      assert!(mask.contains(channel));
    }
  }

  #[test]
  fn default_options_are_sane() {
    let options = RecallOptions::default();
    assert!(options.token_budget > 0);
    assert_eq!(options.channels, ChannelMask::ALL);
    assert_eq!(options.scope, FactScope::All);
    assert!(options.time_range.is_none());
  }

  #[test]
  fn take_hits_limits_and_preserves_order() {
    let ids = [FactId::new(), FactId::new(), FactId::new()];
    let hits = ids
      .iter()
      .map(|id| (*id, 1.0))
      .collect::<Vec<(FactId, f32)>>();

    assert_eq!(take_hits(hits.clone(), 2), vec![ids[0], ids[1]]);
    assert_eq!(take_hits(hits.clone(), 0), Vec::new());
    assert_eq!(take_hits(hits, 99), ids.to_vec());
  }

  #[test]
  fn semantic_seeds_use_reciprocal_rank() {
    let ids = [FactId::new(), FactId::new()];
    let lists = vec![(Channel::Semantic, ids.to_vec())];
    let seeds = semantic_seeds(&lists);

    assert_eq!(seeds.len(), 2);
    assert_eq!(seeds[0].0, ids[0]);
    assert!((seeds[0].1 - 1.0).abs() < 1e-6);
    assert!((seeds[1].1 - 0.5).abs() < 1e-6);

    assert!(semantic_seeds(&[]).is_empty());
    assert!(semantic_seeds(&[(Channel::Bm25, ids.to_vec())]).is_empty());
  }

  #[test]
  fn collect_ranks_merges_channels() {
    let shared = FactId::new();
    let only = FactId::new();
    let lists = vec![
      (Channel::Semantic, vec![shared]),
      (Channel::Bm25, vec![only, shared]),
    ];

    let ranks = collect_ranks(&lists);
    assert_eq!(
      ranks[&shared],
      vec![(Channel::Semantic, 1), (Channel::Bm25, 2)]
    );
    assert_eq!(ranks[&only], vec![(Channel::Bm25, 1)]);
  }
}
