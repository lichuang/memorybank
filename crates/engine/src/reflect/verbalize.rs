//! Verbalizing the behavioral profile Θ (paper §5.2.1–§5.3.1).
//!
//! The numeric profile is turned into one stable sentence so it can be injected
//! into a system message. Each disposition level maps to its own wording, and
//! the bias strength adds one of three modulation clauses. Output depends only on
//! the profile: the same profile always produces byte-identical text, which keeps
//! prompts stable for snapshot tests and caching.

use crate::types::BankProfile;

/// Verbalizes `profile` as one sentence ending in a period.
///
/// The three disposition dimensions each contribute a clause in the order
/// skepticism → literalism → empathy, followed by a bias-strength clause. Wording
/// follows the paper's anchors: 1 = trusting / flexible / detached, 5 =
/// skeptical / literal / empathetic. Values outside `1..=5` are clamped so a
/// malformed profile still yields usable text; call
/// [`BankProfile::validate`](crate::types::BankProfile::validate) before
/// persisting instead of relying on the clamp.
///
/// Bias bands: `β < 0.25` → fact-based, `β < 0.75` → balanced, otherwise strongly
/// preference-conditioned (the paper's 0 / 0.5 / 1 anchors).
pub fn verbalize(profile: &BankProfile) -> String {
  format!(
    "You are {}, {}, and {}. {}",
    skepticism_clause(profile.skepticism),
    literalism_clause(profile.literalism),
    empathy_clause(profile.empathy),
    bias_clause(profile.bias_strength)
  )
}

fn skepticism_clause(level: u8) -> &'static str {
  match clamp_level(level) {
    1 => "generally trusting of what you are told",
    2 => "mostly trusting, though you notice weak support",
    3 => "balanced between trust and scrutiny",
    4 => "inclined to question claims until evidence supports them",
    _ => "highly skeptical and careful about accepting unsupported claims",
  }
}

fn literalism_clause(level: u8) -> &'static str {
  match clamp_level(level) {
    1 => "interpret language very flexibly",
    2 => "read between the lines and infer implicit goals",
    3 => "take wording at face value while allowing reasonable inference",
    4 => "stick closely to the exact wording and explicit instructions",
    _ => "interpret language literally and follow instructions to the letter",
  }
}

fn empathy_clause(level: u8) -> &'static str {
  match clamp_level(level) {
    1 => "stay detached and task-first",
    2 => "keep an impersonal, matter-of-fact tone",
    3 => "balance directness with awareness of the other person",
    4 => "take emotional context into account and soften where it helps",
    _ => "are highly empathetic, using supportive and face-saving language",
  }
}

fn bias_clause(bias: f32) -> &'static str {
  if !bias.is_finite() {
    return "Let the facts drive your reasoning.";
  }
  if bias < 0.25 {
    "Let the facts drive your reasoning, with little stylistic bias."
  } else if bias < 0.75 {
    "Balance the facts with your configured perspective."
  } else {
    "Let your configured perspective strongly shape how you weigh and express things."
  }
}

fn clamp_level(level: u8) -> u8 {
  level.clamp(1, 5)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn profile(skepticism: u8, literalism: u8, empathy: u8, bias: f32) -> BankProfile {
    BankProfile {
      background: "I am a careful assistant.".into(),
      skepticism,
      literalism,
      empathy,
      bias_strength: bias,
    }
  }

  #[test]
  fn matches_the_papers_example_shape() {
    // §5.3.1 example: "generally trusting … interpret language flexibly … highly
    // empathetic".
    let text = verbalize(&profile(1, 1, 5, 0.0));

    assert!(text.starts_with("You are "));
    assert!(text.contains("generally trusting"));
    assert!(text.contains("interpret language very flexibly"));
    assert!(text.contains("highly empathetic"));
    assert!(text.ends_with('.'));
  }

  #[test]
  fn every_level_has_distinct_wording() {
    for (levels, clause) in [
      ((1, 1, 1), skepticism_clause(1)),
      ((2, 2, 2), skepticism_clause(2)),
      ((3, 3, 3), skepticism_clause(3)),
      ((4, 4, 4), skepticism_clause(4)),
      ((5, 5, 5), skepticism_clause(5)),
    ] {
      let text = verbalize(&profile(levels.0, levels.1, levels.2, 0.5));
      assert!(text.contains(clause), "level {:?}: {text}", levels.0);
    }

    let mut clauses = vec![
      skepticism_clause(1),
      skepticism_clause(2),
      skepticism_clause(3),
      skepticism_clause(4),
      skepticism_clause(5),
      literalism_clause(1),
      literalism_clause(2),
      literalism_clause(3),
      literalism_clause(4),
      literalism_clause(5),
      empathy_clause(1),
      empathy_clause(2),
      empathy_clause(3),
      empathy_clause(4),
      empathy_clause(5),
      bias_clause(0.0),
      bias_clause(0.5),
      bias_clause(1.0),
    ];
    clauses.sort_unstable();
    clauses.dedup();
    assert_eq!(clauses.len(), 18, "all level wordings are distinct");
  }

  #[test]
  fn output_is_stable_for_the_same_profile() {
    let profile = profile(4, 2, 5, 0.5);

    assert_eq!(verbalize(&profile), verbalize(&profile));
  }

  #[test]
  fn bias_bands_change_only_the_modulation_clause() {
    let low = verbalize(&profile(3, 3, 3, 0.0));
    let mid = verbalize(&profile(3, 3, 3, 0.5));
    let high = verbalize(&profile(3, 3, 3, 1.0));

    assert!(low.contains("little stylistic bias"));
    assert!(mid.contains("Balance the facts"));
    assert!(high.contains("strongly shape"));

    let core = |text: &str| text.split(". ").next().unwrap_or_default().to_string();
    assert_eq!(core(&low), core(&mid));
    assert_eq!(core(&mid), core(&high));
  }

  #[test]
  fn bias_boundaries_follow_the_paper_anchors() {
    assert!(verbalize(&profile(3, 3, 3, 0.24)).contains("little stylistic bias"));
    assert!(verbalize(&profile(3, 3, 3, 0.25)).contains("Balance the facts"));
    assert!(verbalize(&profile(3, 3, 3, 0.74)).contains("Balance the facts"));
    assert!(verbalize(&profile(3, 3, 3, 0.75)).contains("strongly shape"));
    assert!(verbalize(&profile(3, 3, 3, 1.0)).contains("strongly shape"));
  }

  #[test]
  fn out_of_range_levels_are_clamped() {
    assert_eq!(
      verbalize(&profile(0, 0, 0, 0.5)),
      verbalize(&profile(1, 1, 1, 0.5))
    );
    assert_eq!(
      verbalize(&profile(9, 9, 9, 0.5)),
      verbalize(&profile(5, 5, 5, 0.5))
    );
  }

  #[test]
  fn non_finite_bias_falls_back_to_the_fact_based_clause() {
    assert!(verbalize(&profile(3, 3, 3, f32::NAN)).contains("Let the facts drive"));
  }

  #[test]
  fn extreme_profiles_read_as_coherent_sentences() {
    let permissive = verbalize(&profile(1, 1, 5, 0.0));
    let strict = verbalize(&profile(5, 5, 1, 1.0));

    assert_eq!(permissive.split('.').count(), 3, "one sentence + period");
    assert_eq!(strict.split('.').count(), 3);
    assert_ne!(permissive, strict);
    assert!(!permissive.contains("  "), "no doubled spaces");
    assert!(!strict.contains("  "));
  }
}
