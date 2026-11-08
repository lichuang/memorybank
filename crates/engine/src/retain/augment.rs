//! Time normalization and augmented text.
//!
//! The stored `fact.text` is already the augmented string: `what` (a complete
//! narrative) prefixed with a human-readable time reference (occurrence when
//! known, otherwise the mention time). The same string is fed to the embedder
//! and the reranker, so retrieval never sees a different representation than
//! what is stored (schema §7.A#1).

use chrono::Datelike;

use crate::types::{Chunk, ExtractedFact, Timestamp, from_epoch_millis, to_epoch_millis};

/// Fills the mention time from the chunk's known bounds.
///
/// Only `mentioned_at` is derived (the chunk midpoint, or its only bound).
/// Occurrence bounds are **never** fabricated from the chunk: the chunk window
/// is conversation time, not when the fact happened. A one-sided occurrence
/// stays one-sided (an open interval); `validate_extracted` still rejects
/// reversed or end-only pairs as model errors.
///
/// The wall clock is deliberately **not** used: a fact with no temporal context
/// keeps `None` instead of being stamped with ingestion time.
pub fn normalize_time(extracted: &mut ExtractedFact, chunk: &Chunk) {
  if extracted.mentioned_at.is_none() {
    extracted.mentioned_at = match (chunk.time_start, chunk.time_end) {
      (Some(start), Some(end)) => Some(midpoint(start, end)),
      (Some(start), None) => Some(start),
      (None, Some(end)) => Some(end),
      (None, None) => None,
    };
  }
}

fn midpoint(start: Timestamp, end: Timestamp) -> Timestamp {
  let start = to_epoch_millis(start);
  let end = to_epoch_millis(end);
  let middle = start + (end - start) / 2;
  from_epoch_millis(middle).unwrap_or_else(|| from_epoch_millis(start).expect("start is valid"))
}

/// Formats a timestamp as an English, human-readable UTC date.
///
/// Example: `"Sunday, June 9, 2024"` (day is not zero-padded).
pub fn human_readable(timestamp: Timestamp) -> String {
  format!(
    "{}, {} {}, {}",
    timestamp.format("%A"),
    timestamp.format("%B"),
    timestamp.day(),
    timestamp.year()
  )
}

/// Prepends the human-readable time to a narrative, when a time is known.
///
/// Returns the narrative unchanged when `timestamp` is `None`. Callers must store
/// exactly the returned string: it is both `fact.text` and the embedding input.
pub fn augmented_text(narrative: &str, timestamp: Option<Timestamp>) -> String {
  match timestamp {
    Some(timestamp) => format!("{}. {narrative}", human_readable(timestamp)),
    None => narrative.to_string(),
  }
}

/// Returns the human-readable time reference for a fact.
///
/// Occurrence-first: when both `occurred_start` and `occurred_end` are known,
/// a single date is used for a point event and `From <start> to <end>` for an
/// interval. A one-sided occurrence uses its known bound; only when the
/// occurrence is entirely unknown does it fall back to `mentioned_at`. Returns
/// `None` when no time is known.
pub fn fact_time(extracted: &ExtractedFact) -> Option<String> {
  match (extracted.occurred_start, extracted.occurred_end) {
    (Some(start), Some(end)) if start == end => Some(human_readable(start)),
    (Some(start), Some(end)) => {
      let (start, end) = if start <= end {
        (start, end)
      } else {
        (end, start)
      };
      Some(format!(
        "From {} to {}",
        human_readable(start),
        human_readable(end)
      ))
    }
    (Some(start), None) => Some(human_readable(start)),
    (None, Some(end)) => Some(human_readable(end)),
    (None, None) => extracted.mentioned_at.map(human_readable),
  }
}

/// Builds the stored `fact.text` from an extracted fact.
///
/// The prefix is [`fact_time`] (occurrence date/range when known, otherwise the
/// mention time). The prompt asks the model to weave when/where/who/why into
/// `what`, but that is not guaranteed, so any non-empty dimension not already
/// present is appended in a compact suffix. This keeps the information visible
/// to BM25, vector search, and the reranker instead of losing it. The returned
/// string is exactly what is stored and embedded.
pub fn fact_text(extracted: &ExtractedFact) -> String {
  let narrative = match fact_time(extracted) {
    Some(reference) => format!("{reference}. {}", extracted.what),
    None => extracted.what.clone(),
  };
  let mut missing = Vec::new();

  for (label, value) in [
    ("when", extracted.when.as_str()),
    ("where", extracted.where_.as_str()),
    ("who", extracted.who.as_str()),
    ("why", extracted.why.as_str()),
  ] {
    if !value.trim().is_empty() && !contains_normalized(&narrative, value) {
      missing.push(format!("{label}: {value}"));
    }
  }

  if missing.is_empty() {
    narrative
  } else {
    format!("{narrative} ({})", missing.join("; "))
  }
}

/// Structured extraction dimensions kept for audit and future retrieval.
///
/// Empty fields are omitted; returns `None` when all four are empty.
pub fn fact_attrs(extracted: &ExtractedFact) -> Option<serde_json::Value> {
  let mut attrs = serde_json::Map::new();
  for (key, value) in [
    ("when", extracted.when.as_str()),
    ("where", extracted.where_.as_str()),
    ("who", extracted.who.as_str()),
    ("why", extracted.why.as_str()),
  ] {
    if !value.trim().is_empty() {
      attrs.insert(
        key.to_string(),
        serde_json::Value::String(value.to_string()),
      );
    }
  }

  if attrs.is_empty() {
    None
  } else {
    Some(serde_json::Value::Object(attrs))
  }
}

fn contains_normalized(haystack: &str, needle: &str) -> bool {
  let needle = normalize_whitespace(needle);
  !needle.is_empty() && normalize_whitespace(haystack).contains(&needle)
}

fn normalize_whitespace(value: &str) -> String {
  value
    .split_whitespace()
    .collect::<Vec<_>>()
    .join(" ")
    .to_lowercase()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::timestamp;
  use crate::types::{BankId, ChunkId, ChunkStatus, ExtractedFact, FactType, SourceId};

  fn chunk(time_start: Option<i64>, time_end: Option<i64>) -> Chunk {
    Chunk {
      id: ChunkId::new(),
      bank_id: BankId::new(),
      source_id: SourceId::new(),
      chunk_index: 0,
      text: "text".into(),
      content_hash: "hash".into(),
      status: ChunkStatus::Pending,
      time_start: time_start.map(timestamp),
      time_end: time_end.map(timestamp),
      created_at: timestamp(0),
      updated_at: timestamp(0),
    }
  }

  fn extracted() -> ExtractedFact {
    ExtractedFact {
      what: "Emily joined Google as a software engineer.".into(),
      when: "Saturday, June 9, 2024".into(),
      where_: "Berlin".into(),
      who: "Emily".into(),
      why: "career growth".into(),
      fact_type: FactType::World,
      occurred_start: None,
      occurred_end: None,
      mentioned_at: None,
      entities: Vec::new(),
      causal_relations: Vec::new(),
      context: None,
    }
  }

  #[test]
  fn fills_only_the_mention_time_from_the_chunk() {
    let mut fact = extracted();

    normalize_time(&mut fact, &chunk(Some(10_000), Some(20_000)));

    assert_eq!(fact.occurred_start, None, "occurrence is never fabricated");
    assert_eq!(fact.occurred_end, None, "occurrence is never fabricated");
    assert_eq!(fact.mentioned_at, Some(timestamp(15_000)), "midpoint");
  }

  #[test]
  fn uses_the_only_known_chunk_bound() {
    let mut start_only = extracted();
    normalize_time(&mut start_only, &chunk(Some(10_000), None));
    assert_eq!(start_only.mentioned_at, Some(timestamp(10_000)));
    assert_eq!(start_only.occurred_start, None);
    assert_eq!(start_only.occurred_end, None);

    let mut end_only = extracted();
    normalize_time(&mut end_only, &chunk(None, Some(20_000)));
    assert_eq!(end_only.mentioned_at, Some(timestamp(20_000)));
    assert_eq!(end_only.occurred_start, None);
    assert_eq!(end_only.occurred_end, None);
  }

  #[test]
  fn keeps_supplied_times_untouched() {
    let mut fact = extracted();
    fact.occurred_start = Some(timestamp(5_000));
    fact.mentioned_at = Some(timestamp(7_000));

    normalize_time(&mut fact, &chunk(Some(10_000), Some(20_000)));

    assert_eq!(fact.occurred_start, Some(timestamp(5_000)));
    assert_eq!(
      fact.occurred_end, None,
      "occurrence is not filled from the chunk"
    );
    assert_eq!(fact.mentioned_at, Some(timestamp(7_000)));
  }

  #[test]
  fn never_fabricates_time_without_a_chunk_interval() {
    let mut fact = extracted();

    normalize_time(&mut fact, &chunk(None, None));

    assert_eq!(fact.occurred_start, None);
    assert_eq!(fact.occurred_end, None);
    assert_eq!(fact.mentioned_at, None);
  }

  #[test]
  fn formats_human_readable_dates() {
    assert_eq!(
      human_readable(timestamp(1_717_934_000_000)),
      "Sunday, June 9, 2024"
    );
    assert_eq!(
      human_readable(timestamp(1_704_067_200_000)),
      "Monday, January 1, 2024"
    );
  }

  #[test]
  fn prefixes_only_when_a_time_is_known() {
    assert_eq!(
      augmented_text("Emily joined Google.", Some(timestamp(1_717_934_000_000))),
      "Sunday, June 9, 2024. Emily joined Google."
    );
    assert_eq!(
      augmented_text("Emily joined Google.", None),
      "Emily joined Google."
    );
  }

  #[test]
  fn fact_text_prefixes_time_and_appends_missing_dimensions() {
    let mut fact = extracted();
    normalize_time(
      &mut fact,
      &chunk(Some(1_717_934_000_000), Some(1_717_934_000_000)),
    );

    assert_eq!(
      fact_text(&fact),
      "Sunday, June 9, 2024. Emily joined Google as a software engineer. \
       (when: Saturday, June 9, 2024; where: Berlin; why: career growth)"
    );

    let mut untimed = extracted();
    normalize_time(&mut untimed, &chunk(None, None));
    assert_eq!(
      fact_text(&untimed),
      "Emily joined Google as a software engineer. \
       (when: Saturday, June 9, 2024; where: Berlin; why: career growth)"
    );
  }

  #[test]
  fn fact_text_prefers_occurrence_time_over_mention_time() {
    let mut fact = extracted();
    fact.occurred_start = Some(timestamp(1_741_564_800_000)); // 2025-03-10
    fact.occurred_end = fact.occurred_start;
    fact.mentioned_at = Some(timestamp(1_749_427_200_000)); // 2025-06-09
    fact.when.clear();
    fact.where_.clear();
    fact.who.clear();
    fact.why.clear();

    let text = fact_text(&fact);
    assert!(text.starts_with("Monday, March 10, 2025. "), "text: {text}");
    assert!(!text.contains("June"), "text: {text}");
  }

  #[test]
  fn fact_text_formats_an_occurrence_range() {
    let mut fact = extracted();
    fact.occurred_start = Some(timestamp(1_741_564_800_000)); // 2025-03-10
    fact.occurred_end = Some(timestamp(1_741_737_600_000)); // 2025-03-12
    fact.mentioned_at = None;
    fact.when.clear();
    fact.where_.clear();
    fact.who.clear();
    fact.why.clear();

    let text = fact_text(&fact);
    assert!(
      text.starts_with("From Monday, March 10, 2025 to Wednesday, March 12, 2025. "),
      "text: {text}"
    );
  }

  #[test]
  fn fact_text_falls_back_to_mention_time() {
    let mut fact = extracted();
    fact.occurred_start = None;
    fact.occurred_end = None;
    fact.mentioned_at = Some(timestamp(1_741_564_800_000));
    fact.when.clear();
    fact.where_.clear();
    fact.who.clear();
    fact.why.clear();

    let text = fact_text(&fact);
    assert!(text.starts_with("Monday, March 10, 2025. "), "text: {text}");
  }

  #[test]
  fn fact_text_does_not_duplicate_weaved_dimensions() {
    let mut fact = extracted();
    fact.what =
      "On Saturday, June 9, 2024, Emily joined Google in Berlin because of career growth.".into();
    normalize_time(&mut fact, &chunk(None, None));

    assert_eq!(fact_text(&fact), fact.what);
  }

  #[test]
  fn fact_attrs_keeps_the_structured_dimensions() {
    let attrs = fact_attrs(&extracted()).expect("non-empty attrs");
    assert_eq!(attrs["when"], "Saturday, June 9, 2024");
    assert_eq!(attrs["where"], "Berlin");
    assert_eq!(attrs["who"], "Emily");
    assert_eq!(attrs["why"], "career growth");

    let mut empty = extracted();
    empty.when.clear();
    empty.where_.clear();
    empty.who.clear();
    empty.why.clear();
    assert_eq!(fact_attrs(&empty), None);
  }
}
