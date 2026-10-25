//! Structured-output contracts exchanged with the LLM (paper A.1/A.2/A.3/A.5.1).
//!
//! Prompts are generated from [`schemars::schema_for!`] and responses are
//! deserialized into these types. They are deliberately separate from the row
//! structs: an LLM response is a **proposal**, not a persistence record — the
//! engine validates, resolves, and rewrites it before anything is stored.
//!
//! Leniency rules:
//! - nested detail lists (`entities`, `causal_relations`) may be omitted by the
//!   model and default to empty;
//! - the top-level list of a response (`facts`, `opinions`, `observations`) is
//!   required — its absence means the contract was violated, not "no results".

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{CausalRelation, EntityKind, FactType, Timestamp};

/// One fact proposed by the extraction model.
///
/// The five Ws follow the extraction prompt (paper A.1); the structured fields
/// are the resolved values that land on the fact row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExtractedFact {
  /// What happened, as a maximally verbose narrative statement.
  pub what: String,
  /// When it happened, normalized to natural language.
  pub when: String,
  /// Where it happened; serialized as `where` because `where` is a Rust keyword.
  #[serde(rename = "where")]
  pub where_: String,
  /// Who was involved, with references expanded (e.g. "Emily (the user's roommate)").
  pub who: String,
  /// Why it happened.
  pub why: String,
  /// Memory network this fact belongs to.
  pub fact_type: FactType,
  /// Occurrence interval start in UTC, when the model can resolve it.
  pub occurred_start: Option<Timestamp>,
  /// Occurrence interval end in UTC, when the model can resolve it.
  pub occurred_end: Option<Timestamp>,
  /// Mention time in UTC, when the model can resolve it.
  pub mentioned_at: Option<Timestamp>,
  /// Entity mentions; resolved into canonical entities by the engine.
  #[serde(default)]
  pub entities: Vec<EntityMention>,
  /// Causal references to sibling facts of the same response.
  #[serde(default)]
  pub causal_relations: Vec<CausalRef>,
  /// Optional surrounding context kept for disambiguation.
  pub context: Option<String>,
}

/// An entity mention inside an extracted fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EntityMention {
  /// Raw mention text as it appeared in the source.
  pub text: String,
  /// Optional category hint from the model; resolution may override it.
  pub kind: Option<EntityKind>,
}

/// A causal link between the enclosing fact and a sibling fact.
///
/// References are positional (`target_fact_index`) because sibling fact ids do
/// not exist yet at extraction time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CausalRef {
  /// Index into the enclosing response's `facts` list.
  pub target_fact_index: usize,
  /// Relation from the enclosing fact to the target.
  pub relation: CausalRelation,
  /// Model-reported strength; the engine clamps it before persisting.
  pub strength: f32,
}

/// Extraction response for one chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FactExtractionResponse {
  /// Facts proposed for the chunk; required even when empty.
  pub facts: Vec<ExtractedFact>,
}

/// One opinion proposed by reflection or opinion formation (paper A.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OpinionDraft {
  /// The opinion statement.
  pub opinion: String,
  /// Confidence in [0, 1]; the engine clamps it.
  pub confidence: f32,
  /// Why the model believes it, preserved for auditability.
  pub reasoning: String,
}

/// Reflection response: a user-facing answer plus any new opinions (paper A.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ReflectResponse {
  /// The answer to the reflect query.
  pub answer: String,
  /// Opinions to persist; required even when empty.
  pub opinions: Vec<OpinionDraft>,
}

/// Verdict of the Assess operation (paper A.5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentVerdict {
  /// New evidence strengthens the opinion.
  Reinforce,
  /// New evidence weakens the opinion.
  Weaken,
  /// New evidence contradicts the opinion.
  Contradict,
  /// New evidence is unrelated.
  Neutral,
}

/// Assessment of an opinion against new evidence (paper A.5.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Assessment {
  /// The verdict to apply.
  pub verdict: AssessmentVerdict,
  /// The model's justification, kept for auditability.
  pub reason: String,
}

/// One observation in a synthesized entity profile (paper A.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ObservationDraft {
  /// A single preference-neutral observation sentence.
  pub observation: String,
}

/// Observation synthesis response for one entity (paper A.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ObservationResponse {
  /// Observations to append; required even when empty.
  pub observations: Vec<ObservationDraft>,
}

#[cfg(test)]
mod tests {
  use serde_json::{Value, json};

  use super::*;

  const EXTRACTION_FIXTURE: &str = r#"
  {
    "facts": [
      {
        "what": "Emily started a new job at Google as a software engineer.",
        "when": "in May 2023",
        "where": "Mountain View, California",
        "who": "Emily (the user's roommate)",
        "why": "She accepted an offer after graduating.",
        "fact_type": "world",
        "occurred_start": "2023-05-01T00:00:00Z",
        "occurred_end": null,
        "mentioned_at": "2023-06-10T14:00:00Z"
      }
    ]
  }
  "#;

  #[test]
  fn schema_uses_paper_field_names() {
    let schema = serde_json::to_value(schemars::schema_for!(ExtractedFact)).expect("schema json");
    let properties = schema
      .get("properties")
      .and_then(Value::as_object)
      .expect("object schema");

    for name in ["what", "when", "where", "who", "why"] {
      assert!(properties.contains_key(name), "missing property {name}");
    }
    assert!(!properties.contains_key("where_"));

    let occurred_start = properties
      .get("occurred_start")
      .expect("occurred_start property")
      .to_string();
    assert!(
      occurred_start.contains("date-time"),
      "timestamp schema should be date-time: {occurred_start}"
    );

    let defs = schema
      .get("$defs")
      .and_then(Value::as_object)
      .expect("definitions");
    let fact_type = defs
      .get("FactType")
      .expect("FactType definition")
      .to_string();
    assert!(
      fact_type.contains("\"world\""),
      "fact_type schema should reuse the enum wire names: {fact_type}"
    );
  }

  #[test]
  fn optional_nested_lists_default_to_empty() {
    let response: FactExtractionResponse =
      serde_json::from_str(EXTRACTION_FIXTURE).expect("deserialize fixture");

    assert_eq!(response.facts.len(), 1);
    let fact = &response.facts[0];
    assert!(fact.entities.is_empty());
    assert!(fact.causal_relations.is_empty());
    assert!(fact.context.is_none());
    assert_eq!(fact.occurred_end, None);
  }

  #[test]
  fn where_field_serializes_as_keyword() {
    let fact = ExtractedFact {
      what: "what".into(),
      when: "when".into(),
      where_: "Berlin".into(),
      who: "who".into(),
      why: "why".into(),
      fact_type: FactType::World,
      occurred_start: None,
      occurred_end: None,
      mentioned_at: None,
      entities: vec![EntityMention {
        text: "Berlin".into(),
        kind: Some(EntityKind::Location),
      }],
      causal_relations: vec![CausalRef {
        target_fact_index: 0,
        relation: CausalRelation::Causes,
        strength: 0.5,
      }],
      context: None,
    };

    let encoded = serde_json::to_value(&fact).expect("serialize fact");
    assert_eq!(encoded.get("where"), Some(&json!("Berlin")));
    assert!(encoded.get("where_").is_none());
  }

  #[test]
  fn top_level_list_is_required() {
    let error = serde_json::from_str::<FactExtractionResponse>("{}").expect_err("missing facts");
    assert!(error.to_string().contains("facts"));

    let error = serde_json::from_str::<ReflectResponse>(r#"{"answer": "yes"}"#)
      .expect_err("missing opinions");
    assert!(error.to_string().contains("opinions"));
  }

  #[test]
  fn wrong_types_are_rejected() {
    for fixture in [
      r#"{"facts": {}}"#,
      r#"{"facts": [{"what": "w", "when": "w", "where": "w", "who": "w", "why": "w", "fact_type": "nonsense"}]}"#,
      r#"{"facts": [{"what": "w", "when": "w", "where": "w", "who": "w", "why": "w", "fact_type": "world", "occurred_start": "not-a-date"}]}"#,
    ] {
      assert!(
        serde_json::from_str::<FactExtractionResponse>(fixture).is_err(),
        "fixture should be rejected: {fixture}"
      );
    }
  }

  #[test]
  fn assessment_verdicts_use_snake_case() {
    for (verdict, wire) in [
      (AssessmentVerdict::Reinforce, "reinforce"),
      (AssessmentVerdict::Weaken, "weaken"),
      (AssessmentVerdict::Contradict, "contradict"),
      (AssessmentVerdict::Neutral, "neutral"),
    ] {
      let json = serde_json::to_string(&verdict).expect("serialize verdict");
      assert_eq!(json, format!("\"{wire}\""));
      let back: AssessmentVerdict = serde_json::from_str(&json).expect("deserialize verdict");
      assert_eq!(back, verdict);
    }
  }
}
