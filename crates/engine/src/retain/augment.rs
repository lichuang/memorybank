//! Time normalization and augmented text.
//!
//! The stored `fact.text` is already the augmented string: `what` (a complete
//! narrative) prefixed with the human-readable mention time. The same string is
//! fed to the embedder and the reranker, so retrieval never sees a different
//! representation than what is stored (schema §7.A#1).

use chrono::Datelike;

use crate::types::{Chunk, ExtractedFact, Timestamp, from_epoch_millis, to_epoch_millis};

/// Fills missing time fields from the chunk's known bounds.
///
/// - `occurred_start` / `occurred_end` default to the chunk interval ends;
/// - `mentioned_at` defaults to the chunk interval midpoint (or its only bound).
///
/// The wall clock is deliberately **not** used: a fact with no temporal context
/// keeps `None` instead of being stamped with ingestion time.
pub fn normalize_time(extracted: &mut ExtractedFact, chunk: &Chunk) {
  if extracted.occurred_start.is_none() {
    extracted.occurred_start = chunk.time_start;
  }
  if extracted.occurred_end.is_none() {
    extracted.occurred_end = chunk.time_end;
  }
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

/// Builds the stored `fact.text` from an extracted fact.
///
/// The narrative is `what` (the prompt requires it to weave in the other four
/// dimensions); the prefix uses `mentioned_at`. Call
/// [`normalize_time`] first so a missing `mentioned_at` can be filled from the
/// chunk.
pub fn fact_text(extracted: &ExtractedFact) -> String {
  augmented_text(&extracted.what, extracted.mentioned_at)
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
  fn fills_missing_times_from_the_chunk_interval() {
    let mut fact = extracted();

    normalize_time(&mut fact, &chunk(Some(10_000), Some(20_000)));

    assert_eq!(fact.occurred_start, Some(timestamp(10_000)));
    assert_eq!(fact.occurred_end, Some(timestamp(20_000)));
    assert_eq!(fact.mentioned_at, Some(timestamp(15_000)), "midpoint");
  }

  #[test]
  fn uses_the_only_known_chunk_bound() {
    let mut start_only = extracted();
    normalize_time(&mut start_only, &chunk(Some(10_000), None));
    assert_eq!(start_only.mentioned_at, Some(timestamp(10_000)));
    assert_eq!(start_only.occurred_end, None, "no end is invented");

    let mut end_only = extracted();
    normalize_time(&mut end_only, &chunk(None, Some(20_000)));
    assert_eq!(end_only.mentioned_at, Some(timestamp(20_000)));
    assert_eq!(end_only.occurred_start, None);
  }

  #[test]
  fn keeps_supplied_times_untouched() {
    let mut fact = extracted();
    fact.occurred_start = Some(timestamp(5_000));
    fact.mentioned_at = Some(timestamp(7_000));

    normalize_time(&mut fact, &chunk(Some(10_000), Some(20_000)));

    assert_eq!(fact.occurred_start, Some(timestamp(5_000)));
    assert_eq!(
      fact.occurred_end,
      Some(timestamp(20_000)),
      "only the end was missing"
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
  fn fact_text_uses_what_and_mentioned_at() {
    let mut fact = extracted();
    normalize_time(
      &mut fact,
      &chunk(Some(1_717_934_000_000), Some(1_717_934_000_000)),
    );

    assert_eq!(
      fact_text(&fact),
      "Sunday, June 9, 2024. Emily joined Google as a software engineer."
    );

    let mut untimed = extracted();
    normalize_time(&mut untimed, &chunk(None, None));
    assert_eq!(fact_text(&untimed), untimed.what);
  }
}
