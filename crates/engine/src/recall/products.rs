//! Recall products: the externally returned evidence plus per-channel provenance.
//!
//! `RecallResult` is what the caller (and later reflect) consumes; it also
//! records which channels produced each fact so multi-channel agreement is
//! inspectable without parsing text or re-running retrieval.

use serde::{Deserialize, Serialize};

use crate::types::{Fact, FactId};

/// Retrieval channel that produced a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
  /// Vector similarity.
  Semantic,
  /// Lexical BM25.
  Bm25,
  /// Spreading activation over the memory graph.
  Graph,
  /// Temporal proximity to a parsed query range.
  Temporal,
}

impl Channel {
  /// Stable lowercase wire name, usable as a label/tag.
  pub fn as_str(&self) -> &'static str {
    match self {
      Self::Semantic => "semantic",
      Self::Bm25 => "bm25",
      Self::Graph => "graph",
      Self::Temporal => "temporal",
    }
  }
}

/// One returned fact with its fused score and per-channel rank evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecalledFact {
  /// The fact row.
  pub fact: Fact,
  /// Final score used for ordering (post-fusion, post-rerank).
  pub score: f32,
  /// Channels that retrieved this fact, with the 1-based rank each assigned.
  ///
  /// Ordering is by channel (semantic, bm25, graph, temporal) so the value is
  /// deterministic; a fact found by several channels appears several times.
  pub channels: Vec<(Channel, u32)>,
}

/// The full recall product returned to the caller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallResult {
  /// Packed evidence, best first.
  pub facts: Vec<RecalledFact>,
  /// Token estimate consumed by the packed facts.
  pub tokens_used: usize,
}

impl RecallResult {
  /// Returns the ids of the packed facts, in order.
  pub fn ids(&self) -> Vec<FactId> {
    self.facts.iter().map(|recalled| recalled.fact.id).collect()
  }

  /// Returns the channels that produced `id`, or an empty list when it is not
  /// part of this result.
  pub fn channels_of(&self, id: FactId) -> Vec<(Channel, u32)> {
    self
      .facts
      .iter()
      .find(|recalled| recalled.fact.id == id)
      .map(|recalled| recalled.channels.clone())
      .unwrap_or_default()
  }

  /// Whether `id` was retrieved by at least two channels.
  pub fn is_multi_channel(&self, id: FactId) -> bool {
    self.channels_of(id).len() >= 2
  }
}

/// Orders channel evidence canonically: semantic, bm25, graph, temporal.
pub fn sort_channels(channels: &mut Vec<(Channel, u32)>) {
  channels.sort_by(|left, right| channel_rank(left.0).cmp(&channel_rank(right.0)));
  channels.dedup_by_key(|(channel, _)| *channel);
}

fn channel_rank(channel: Channel) -> u8 {
  match channel {
    Channel::Semantic => 0,
    Channel::Bm25 => 1,
    Channel::Graph => 2,
    Channel::Temporal => 3,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::timestamp;
  use crate::types::{BankId, FactType, OccurrenceInterval};

  fn fact(text: &str) -> Fact {
    Fact {
      id: FactId::new(),
      bank_id: BankId::new(),
      fact_type: FactType::World,
      text: text.into(),
      embedding: None,
      chunk_id: None,
      occurred: OccurrenceInterval::Unknown,
      mentioned_at: None,
      confidence: None,
      access_count: 0,
      attrs: None,
      created_at: timestamp(0),
      updated_at: timestamp(0),
    }
  }

  #[test]
  fn channel_wire_names_are_stable() {
    for (channel, wire) in [
      (Channel::Semantic, "semantic"),
      (Channel::Bm25, "bm25"),
      (Channel::Graph, "graph"),
      (Channel::Temporal, "temporal"),
    ] {
      assert_eq!(channel.as_str(), wire);
      let json = serde_json::to_string(&channel).expect("serialize channel");
      assert_eq!(json, format!("\"{wire}\""));
    }
  }

  #[test]
  fn channel_evidence_is_sorted_and_deduplicated() {
    let mut channels = vec![
      (Channel::Temporal, 4),
      (Channel::Semantic, 1),
      (Channel::Semantic, 9),
      (Channel::Graph, 2),
    ];
    sort_channels(&mut channels);

    assert_eq!(
      channels,
      vec![
        (Channel::Semantic, 1),
        (Channel::Graph, 2),
        (Channel::Temporal, 4)
      ]
    );
  }

  #[test]
  fn result_accessors_expose_overlap() {
    let first = fact("first");
    let second = fact("second");
    let result = RecallResult {
      facts: vec![
        RecalledFact {
          fact: first.clone(),
          score: 1.5,
          channels: vec![(Channel::Semantic, 1), (Channel::Bm25, 2)],
        },
        RecalledFact {
          fact: second.clone(),
          score: 1.0,
          channels: vec![(Channel::Graph, 1)],
        },
      ],
      tokens_used: 7,
    };

    assert_eq!(result.ids(), vec![first.id, second.id]);
    assert_eq!(
      result.channels_of(first.id),
      vec![(Channel::Semantic, 1), (Channel::Bm25, 2)]
    );
    assert!(result.is_multi_channel(first.id));
    assert!(!result.is_multi_channel(second.id));
    assert!(result.channels_of(FactId::new()).is_empty());
  }

  #[test]
  fn result_round_trips_through_json() {
    let result = RecallResult {
      facts: vec![RecalledFact {
        fact: fact("a"),
        score: 0.75,
        channels: vec![(Channel::Temporal, 1)],
      }],
      tokens_used: 3,
    };

    let encoded = serde_json::to_string(&result).expect("serialize result");
    let decoded: RecallResult = serde_json::from_str(&encoded).expect("deserialize result");
    assert_eq!(decoded, result);
  }
}
