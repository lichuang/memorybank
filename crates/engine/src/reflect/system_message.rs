//! System-message assembly for reflect (paper §5.4.2, step 2).
//!
//! The message carries the three pieces the paper names — the bank's name `n`,
//! its background `h`, and the verbalized profile φ(Θ) — plus a numbered evidence
//! block so the model can cite facts by index. It is a pure function of its
//! inputs, which keeps prompts byte-stable for snapshot tests.

use crate::recall::RecalledFact;
use crate::reflect::verbalize;
use crate::types::Bank;

/// Limits for the evidence block.
///
/// Evidence is already token-packed by recall; these caps are a second guard for
/// callers that assemble a message from a larger candidate set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMessageConfig {
  /// Maximum evidence items included. `0` includes none.
  pub max_evidence: usize,
  /// Maximum characters in the evidence block. `0` disables the cap.
  pub max_chars: usize,
}

impl Default for SystemMessageConfig {
  fn default() -> Self {
    Self {
      max_evidence: 24,
      max_chars: 6_000,
    }
  }
}

/// Builds the reflect system message.
///
/// Layout: bank name, background, verbalized profile, a citation instruction, the
/// numbered evidence block, and a final "say so instead of guessing" guard.
///
/// Evidence is included **whole** and in order: the message never truncates a
/// fact's text, because a partial quote would read as if the memory said
/// something it did not. When the caps exclude items, the block reports how many
/// were omitted so the model knows its view is partial. With no evidence at all,
/// the block explicitly states that no relevant memories were found.
pub fn system_message(bank: &Bank, evidence: &[RecalledFact], cfg: &SystemMessageConfig) -> String {
  let background = bank.profile.background.trim();
  let background = if background.is_empty() {
    "(no background provided)"
  } else {
    background
  };

  let mut message = String::new();
  message.push_str(&format!(
    "You are the memory agent for bank \"{}\".\n\n",
    bank.name
  ));
  message.push_str("Background:\n");
  message.push_str(background);
  message.push_str("\n\n");
  message.push_str(&verbalize(&bank.profile));
  message.push('\n');
  message.push_str(
    "\nUse the numbered memories below as evidence. Cite them by number when they support your answer.\n\n",
  );

  let (block, omitted) = evidence_block(evidence, cfg);
  message.push_str("Relevant memories:\n");
  message.push_str(&block);
  if omitted > 0 {
    message.push_str(&format!(
      "\n[{omitted} more {} omitted]\n",
      if omitted == 1 { "memory" } else { "memories" }
    ));
  }
  message.push('\n');
  message
    .push_str("If these memories do not contain enough information, say so instead of guessing.");

  message
}

/// Renders the numbered block and returns how many items were left out.
fn evidence_block(evidence: &[RecalledFact], cfg: &SystemMessageConfig) -> (String, usize) {
  if evidence.is_empty() {
    return ("(none — no relevant memories were found.)".to_string(), 0);
  }

  let limit = cfg.max_evidence.min(evidence.len());
  let mut block = String::new();
  let mut included = 0_usize;

  for (index, recalled) in evidence.iter().take(limit).enumerate() {
    let line = format!("[{}] {}\n", index + 1, recalled.fact.text.trim());
    // Stop at the first item that would overflow: skipping ahead would promote
    // lower-ranked evidence, the same rule budget packing follows.
    if cfg.max_chars > 0 && block.len() + line.len() > cfg.max_chars {
      break;
    }
    block.push_str(&line);
    included += 1;
  }

  (block, evidence.len() - included)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::recall::Channel;
  use crate::testing::timestamp;
  use crate::types::{BankId, BankProfile, Fact, FactId, FactType, OccurrenceInterval};

  fn bank(name: &str, background: &str, bias: f32) -> Bank {
    Bank {
      id: BankId::new(),
      name: name.into(),
      profile: BankProfile {
        background: background.into(),
        skepticism: 2,
        literalism: 2,
        empathy: 4,
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

  #[test]
  fn carries_name_background_and_verbalized_profile() {
    let mut bank = bank("primary", "I am a careful assistant.", 0.0);
    bank.profile.skepticism = 1;
    bank.profile.empathy = 5;
    let message = system_message(
      &bank,
      &[recalled("Emily works at Google.")],
      &Default::default(),
    );

    assert!(message.contains("bank \"primary\""));
    assert!(message.contains("I am a careful assistant."));
    assert!(message.contains("generally trusting"));
    assert!(message.contains("highly empathetic"));

    let profile = verbalize(&bank.profile);
    assert!(
      message.contains(&profile),
      "the profile sentence appears verbatim"
    );
  }

  #[test]
  fn numbers_evidence_in_order() {
    let bank = bank("primary", "background", 0.5);
    let evidence = [
      recalled("first memory"),
      recalled("second memory"),
      recalled("third memory"),
    ];

    let message = system_message(&bank, &evidence, &Default::default());

    let first = message.find("[1] first memory").expect("first");
    let second = message.find("[2] second memory").expect("second");
    let third = message.find("[3] third memory").expect("third");
    assert!(first < second && second < third);
  }

  #[test]
  fn empty_evidence_states_that_nothing_was_found() {
    let bank = bank("primary", "background", 0.5);
    let message = system_message(&bank, &[], &Default::default());

    assert!(message.contains("no relevant memories were found"));
    assert!(!message.contains("[1]"));
  }

  #[test]
  fn caps_whole_items_and_reports_omissions() {
    let bank = bank("primary", "background", 0.5);
    let evidence: Vec<RecalledFact> = (0..5)
      .map(|index| recalled(&format!("memory number {index}")))
      .collect();

    let cfg = SystemMessageConfig {
      max_evidence: 2,
      max_chars: 0,
    };
    let message = system_message(&bank, &evidence, &cfg);

    assert!(message.contains("[1] memory number 0"));
    assert!(message.contains("[2] memory number 1"));
    assert!(!message.contains("memory number 2"));
    assert!(message.contains("3 more memories omitted"));
  }

  #[test]
  fn char_cap_stops_at_the_first_overflow_without_backfilling() {
    let bank = bank("primary", "background", 0.5);
    let evidence = vec![
      recalled("short"),
      recalled(&"x".repeat(200)),
      recalled("tiny"),
    ];

    let cfg = SystemMessageConfig {
      max_evidence: 10,
      max_chars: 100,
    };
    let message = system_message(&bank, &evidence, &cfg);

    assert!(message.contains("[1] short"));
    assert!(
      !message.contains("[2] "),
      "the oversized item ends the block"
    );
    assert!(
      !message.contains("tiny"),
      "smaller later items are not promoted"
    );
    assert!(message.contains("2 more memories omitted"));
  }

  #[test]
  fn a_single_omission_uses_the_singular_form() {
    let bank = bank("primary", "background", 0.5);
    let evidence = vec![recalled("first"), recalled("second")];
    let cfg = SystemMessageConfig {
      max_evidence: 1,
      max_chars: 0,
    };

    let message = system_message(&bank, &evidence, &cfg);

    assert!(message.contains("1 more memory omitted"));
  }

  #[test]
  fn empty_background_is_marked_explicitly() {
    let bank = bank("primary", "   ", 0.5);
    let message = system_message(&bank, &[], &Default::default());

    assert!(message.contains("(no background provided)"));
  }

  #[test]
  fn output_is_stable_and_ends_with_the_guard() {
    let bank = bank("primary", "background", 1.0);
    let evidence = vec![recalled("a fact")];

    let first = system_message(&bank, &evidence, &Default::default());
    let second = system_message(&bank, &evidence, &Default::default());

    assert_eq!(first, second);
    assert!(first.trim_end().ends_with("say so instead of guessing."));
  }

  #[test]
  fn bias_changes_the_profile_sentence_not_the_evidence() {
    let evidence = vec![recalled("shared evidence")];
    let low = system_message(
      &bank("primary", "background", 0.0),
      &evidence,
      &Default::default(),
    );
    let high = system_message(
      &bank("primary", "background", 1.0),
      &evidence,
      &Default::default(),
    );

    assert!(low.contains("little stylistic bias"));
    assert!(high.contains("strongly shape"));
    assert!(low.contains("[1] shared evidence"));
    assert!(high.contains("[1] shared evidence"));
  }
}
