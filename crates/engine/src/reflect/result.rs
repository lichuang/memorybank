//! The reflect product: answer, formed opinions, and the evidence behind them.
//!
//! The caller must be able to relate the answer to the memories that supported it
//! without parsing text. The link is structural: evidence is numbered exactly as
//! in the system message (`[1]`, `[2]`, …), so a citation number in the answer
//! maps to a fact id through [`ReflectResult::evidence_id`].

use serde::{Deserialize, Serialize};

use crate::recall::RecallResult;
use crate::types::FactId;

/// Everything one reflect call produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflectResult {
  /// The user-facing answer.
  pub answer: String,
  /// Ids of the opinion rows written for this call, in draft order.
  pub opinions_formed: Vec<FactId>,
  /// The recall product the answer was grounded in.
  pub recalled: RecallResult,
}

impl ReflectResult {
  /// Fact ids of the evidence, best first.
  pub fn evidence_ids(&self) -> Vec<FactId> {
    self.recalled.ids()
  }

  /// Maps a 1-based citation number (as used in the system message and the
  /// answer) to the fact it denotes.
  pub fn evidence_id(&self, number: usize) -> Option<FactId> {
    number
      .checked_sub(1)
      .and_then(|index| self.recalled.facts.get(index))
      .map(|recalled| recalled.fact.id)
  }

  /// Maps a fact id back to its 1-based citation number.
  pub fn evidence_number(&self, id: FactId) -> Option<usize> {
    self
      .recalled
      .facts
      .iter()
      .position(|recalled| recalled.fact.id == id)
      .map(|index| index + 1)
  }

  /// Whether the call wrote at least one opinion.
  pub fn formed_any_opinion(&self) -> bool {
    !self.opinions_formed.is_empty()
  }

  /// Whether any evidence was available for this answer.
  pub fn has_evidence(&self) -> bool {
    !self.recalled.facts.is_empty()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::recall::{Channel, RecalledFact};
  use crate::testing::timestamp;
  use crate::types::{BankId, Fact, FactType, OccurrenceInterval};

  fn recalled(text: &str) -> RecalledFact {
    RecalledFact {
      fact: Fact {
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
      },
      score: 1.0,
      channels: vec![(Channel::Semantic, 1)],
    }
  }

  fn result(evidence: Vec<RecalledFact>, opinions: Vec<FactId>) -> ReflectResult {
    ReflectResult {
      answer: "Alice is reliable [1].".into(),
      opinions_formed: opinions,
      recalled: RecallResult {
        facts: evidence,
        tokens_used: 12,
      },
    }
  }

  #[test]
  fn citation_numbers_map_to_fact_ids() {
    let evidence = vec![recalled("first"), recalled("second")];
    let first = evidence[0].fact.id;
    let second = evidence[1].fact.id;
    let result = result(evidence, Vec::new());

    assert_eq!(result.evidence_id(1), Some(first));
    assert_eq!(result.evidence_id(2), Some(second));
    assert_eq!(result.evidence_id(0), None);
    assert_eq!(result.evidence_id(3), None);

    assert_eq!(result.evidence_number(first), Some(1));
    assert_eq!(result.evidence_number(second), Some(2));
    assert_eq!(result.evidence_number(FactId::new()), None);
  }

  #[test]
  fn evidence_round_trips_between_number_and_id() {
    let evidence = vec![recalled("a"), recalled("b"), recalled("c")];
    let result = result(evidence, Vec::new());

    for number in 1..=3 {
      let id = result.evidence_id(number).expect("id");
      assert_eq!(result.evidence_number(id), Some(number));
    }
  }

  #[test]
  fn empty_evidence_is_reported_not_guessed() {
    let result = result(Vec::new(), Vec::new());

    assert!(!result.has_evidence());
    assert!(result.evidence_id(1).is_none());
    assert!(result.evidence_ids().is_empty());
  }

  #[test]
  fn formed_opinions_are_exposed_as_ids() {
    let first = FactId::new();
    let result = result(vec![recalled("a")], vec![first]);

    assert!(result.formed_any_opinion());
    assert_eq!(result.opinions_formed, vec![first]);
    assert!(result.has_evidence());
  }

  #[test]
  fn result_keeps_the_answer_verbatim_and_round_trips() {
    let result = result(vec![recalled("a")], vec![FactId::new()]);

    assert_eq!(result.answer, "Alice is reliable [1].");

    let encoded = serde_json::to_string(&result).expect("serialize");
    let decoded: ReflectResult = serde_json::from_str(&encoded).expect("deserialize");
    assert_eq!(decoded, result);
  }
}
