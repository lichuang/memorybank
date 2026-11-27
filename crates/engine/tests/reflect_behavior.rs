//! Behavioral tests: the profile must be observable in the prompt the model
//! actually receives, not just in the verbalization helper.
//!
//! `FakeLlm` records every request, so these tests assert on the system message
//! and user prompt after a full `system_message → reflect_llm` round trip.

use engine::recall::{Channel, RecallResult, RecalledFact};
use engine::reflect::{ReflectConfig, SystemMessageConfig, reflect_llm, system_message, verbalize};
use engine::testing::{FakeLlm, timestamp};
use engine::types::{Bank, BankId, BankProfile, Fact, FactId, FactType, OccurrenceInterval};
use serde_json::json;

fn bank(name: &str, skepticism: u8, literalism: u8, empathy: u8, bias: f32) -> Bank {
  Bank {
    id: BankId::new(),
    name: name.into(),
    profile: BankProfile {
      background: "I am a careful assistant.".into(),
      skepticism,
      literalism,
      empathy,
      bias_strength: bias,
    },
    created_at: timestamp(0),
    updated_at: timestamp(0),
  }
}

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

fn evidence() -> Vec<RecalledFact> {
  vec![
    recalled("Alice shipped the migration on time."),
    recalled("Bob reviewed the design doc."),
  ]
}

fn canned_response() -> serde_json::Value {
  json!({
    "answer": "Alice shipped the migration on time [1].",
    "opinions": [
      { "opinion": "I think Alice is dependable", "confidence": 0.7, "reasoning": "delivery record" }
    ]
  })
}

/// Runs the two-step reflect prompt path and returns the recorded request.
async fn prompt_for(profile_bank: &Bank, bias_label: &str) -> (String, String) {
  let llm = FakeLlm::new();
  llm.push("reflect", canned_response());

  let system = system_message(profile_bank, &evidence(), &SystemMessageConfig::default());
  let result = reflect_llm(
    &llm,
    system,
    "Is Alice dependable?",
    &ReflectConfig::default(),
  )
  .await
  .expect("reflect");
  assert!(!result.answer.is_empty(), "{bias_label}");

  let calls = llm.calls();
  assert_eq!(calls.len(), 1);
  let system = calls[0].system.clone().expect("system message");
  let user = calls[0].user.clone();
  (system, user)
}

#[tokio::test]
async fn high_and_low_skepticism_produce_different_system_prompts() {
  let low = bank("primary", 1, 3, 3, 0.5);
  let high = bank("primary", 5, 3, 3, 0.5);

  let (low_system, _) = prompt_for(&low, "low").await;
  let (high_system, _) = prompt_for(&high, "high").await;

  assert_ne!(low_system, high_system);
  assert!(low_system.contains(&verbalize(&low.profile)));
  assert!(high_system.contains(&verbalize(&high.profile)));
  assert!(high_system.contains("skeptical"));
}

#[tokio::test]
async fn literalism_and_empathy_also_show_up_in_the_prompt() {
  let literal = bank("primary", 3, 5, 1, 0.5);
  let flexible = bank("primary", 3, 1, 5, 0.5);

  let (literal_system, _) = prompt_for(&literal, "literal").await;
  let (flexible_system, _) = prompt_for(&flexible, "flexible").await;

  assert!(literal_system.contains("literally"));
  assert!(literal_system.contains("detached"));
  assert!(flexible_system.contains("flexibly"));
  assert!(flexible_system.contains("highly empathetic"));
  assert_ne!(literal_system, flexible_system);
}

#[tokio::test]
async fn bias_changes_the_modulation_clause_but_not_the_evidence() {
  let neutral = bank("primary", 3, 3, 3, 0.0);
  let strong = bank("primary", 3, 3, 3, 1.0);

  let (neutral_system, neutral_user) = prompt_for(&neutral, "beta0").await;
  let (strong_system, strong_user) = prompt_for(&strong, "beta1").await;

  assert!(neutral_system.contains("little stylistic bias"));
  assert!(strong_system.contains("strongly shape"));
  assert_ne!(neutral_system, strong_system);

  // Evidence and the question are identical: bias shapes wording, not data.
  for line in [
    "[1] Alice shipped the migration on time.",
    "[2] Bob reviewed the design doc.",
  ] {
    assert!(neutral_system.contains(line));
    assert!(strong_system.contains(line));
  }
  assert_eq!(neutral_user, strong_user);
  assert!(neutral_user.contains("Is Alice dependable?"));
}

#[tokio::test]
async fn evidence_numbering_matches_recall_order() {
  let profile = bank("primary", 3, 3, 3, 0.5);
  let evidence = evidence();
  let first = evidence[0].fact.id;
  let second = evidence[1].fact.id;
  let system = system_message(&profile, &evidence, &SystemMessageConfig::default());

  let first_at = system.find("[1] Alice").expect("first");
  let second_at = system.find("[2] Bob").expect("second");
  assert!(first_at < second_at);

  // The numbering is the contract P3-6 resolves citations through.
  let result = RecallResult {
    facts: evidence,
    tokens_used: 0,
  };
  assert_eq!(result.ids(), vec![first, second]);
}

#[test]
fn verbalization_is_stable_across_repeated_calls() {
  for profile in [
    bank("a", 1, 1, 1, 0.0),
    bank("b", 3, 3, 3, 0.5),
    bank("c", 5, 5, 5, 1.0),
  ] {
    assert_eq!(verbalize(&profile.profile), verbalize(&profile.profile));
  }
}

#[test]
fn the_request_uses_the_reflect_schema_and_prompt() {
  let runtime = tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()
    .expect("runtime");

  runtime.block_on(async {
    let llm = FakeLlm::new();
    llm.push("reflect", canned_response());
    let profile = bank("primary", 4, 2, 5, 0.5);
    let system = system_message(&profile, &evidence(), &SystemMessageConfig::default());
    reflect_llm(&llm, system, "q", &ReflectConfig::default())
      .await
      .expect("reflect");

    let calls = llm.calls();
    let request = &calls[0];
    assert_eq!(request.schema_name, "reflect");
    assert_eq!(request.temperature, ReflectConfig::default().temperature);
    assert!(request.user.starts_with("QUESTION:\nq"));
    assert!(request.user.contains("first person"));
    assert!(request.schema.get("properties").is_some());
  });
}
