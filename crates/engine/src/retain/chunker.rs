//! Chunker: lossless slices of the canonical text.
//!
//! Strategy (paper §4.1.2 "coarse-grained chunking"):
//!
//! 1. **Session alignment first** — group turns by session id, or by a
//!    timestamp gap when ids are absent; never split mid-turn unless the turn
//!    itself exceeds `max_tokens`.
//! 2. **Window fallback** — documents (no spans) and oversized turns are split
//!    at `max_tokens`, always on UTF-8 char boundaries.
//! 3. **Small-chunk merge** — a chunk smaller than `min_tokens` is merged into
//!    the previous one when the result still fits `max_tokens`.
//!
//! Chunk text is always a verbatim slice of the canonical text, so re-assembling
//! the slices reproduces the payload (minus whitespace-only pieces).

use std::ops::Range;

use chrono::Duration;
use sha2::{Digest, Sha256};

use super::normalize::{NormalizedPayload, TurnSpan};
use crate::tokens::TokenEstimator;
use crate::types::Timestamp;

/// Chunking parameters. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkConfig {
  /// A gap strictly larger than this starts a new session when the caller did
  /// not supply `RawTurn::session` ids. When session ids are present and differ,
  /// they win and this value is ignored.
  pub session_gap: Duration,
  /// Soft upper bound in estimated tokens for one chunk; oversized turns and
  /// documents are split at this size. Values below 1 are clamped to 1.
  pub max_tokens: usize,
  /// Soft lower bound: a chunk smaller than this is merged into the previous
  /// chunk when the result still fits `max_tokens` (possibly crossing a session
  /// boundary). Zero disables merging.
  pub min_tokens: usize,
}

/// One chunk produced by [`chunk`], before ids are assigned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawChunk {
  /// Position within the source.
  pub index: u32,
  /// Verbatim slice of the canonical text.
  pub text: String,
  /// Hash of the whitespace-normalized text; the idempotency key.
  pub content_hash: String,
  /// Earliest turn time covered, when known.
  pub time_start: Option<Timestamp>,
  /// Latest turn time covered, when known.
  pub time_end: Option<Timestamp>,
}

/// Splits a normalized payload into chunks.
///
/// Strategy, in order:
///
/// 1. **Session alignment** — turns are grouped by `session`, or by a timestamp
///    gap larger than [`ChunkConfig::session_gap`] when ids are absent.
/// 2. **Window fallback** — documents (no spans) and turns larger than
///    [`ChunkConfig::max_tokens`] are sliced as windows. Windows are cut on
///    UTF-8 char boundaries, never inside a character.
/// 3. **Small-chunk merge** — a chunk below [`ChunkConfig::min_tokens`] is
///    merged into the previous chunk when the result still fits `max_tokens`.
///
/// # Guarantees
///
/// - **Lossless**: each `text` is a byte slice of `payload.text`; concatenating
///   the chunks reproduces the text except for whitespace-only pieces, which are
///   dropped.
/// - **Deterministic**: the same payload, configuration, and estimator always
///   produce the same chunks.
/// - **Non-empty**: no chunk is empty or whitespace-only.
/// - **Time bounds**: `time_start`/`time_end` are the min/max turn timestamps
///   covered by the chunk; documents and untimed turns yield `None`.
///
/// An empty or whitespace-only payload produces no chunks.
pub fn chunk(
  payload: &NormalizedPayload,
  cfg: &ChunkConfig,
  estimator: &dyn TokenEstimator,
) -> Vec<RawChunk> {
  if payload.text.trim().is_empty() {
    return Vec::new();
  }

  let max_tokens = cfg.max_tokens.max(1);
  let pending = if payload.spans.is_empty() {
    document_chunks(&payload.text, max_tokens, estimator)
  } else {
    // Turn spans cover only the turn text; extend them to whole lines first so
    // chunk boundaries fall between turns instead of inside a `[ts] speaker: `
    // prefix (see `build_units`).
    let units = build_units(&payload.text, &payload.spans);
    let mut merged: Vec<PendingChunk> = Vec::new();
    // Split by session first: `group_sessions` only looks at metadata (a change
    // of session id, or a timestamp gap above `session_gap`) and never at size,
    // so every group is a whole-turn slice of the text.
    for group in group_sessions(units, cfg.session_gap) {
      // Then split each session by budget: `chunk_group` greedily packs whole
      // units up to `max_tokens` and only cuts inside a turn when that turn
      // alone is too large. Ranges stay in the original text coordinates.
      for piece in chunk_group(&payload.text, &group, max_tokens, estimator) {
        // Consecutive sessions may still merge below when the new piece is tiny.
        // Small-chunk merge: when this piece is below `min_tokens` and absorbing
        // it into the previous chunk still fits `max_tokens`, extend the previous
        // chunk instead of emitting a tiny standalone chunk (fewer model calls,
        // fewer low-value retrieval units). This may cross a session boundary on
        // purpose; see `ChunkConfig::min_tokens`.
        if let Some(previous) = merged.last_mut() {
          let small = estimator.estimate(&payload.text[piece.range.clone()]) < cfg.min_tokens;
          // Pieces arrive in text order, so `previous.start..piece.end` is the
          // contiguous union of both slices (plus any whitespace between them).
          let combined = previous.range.start..piece.range.end;
          if small && estimator.estimate(&payload.text[combined.clone()]) <= max_tokens {
            previous.range = combined;
            // The merged chunk now covers both pieces' time window. `time_start`
            // is inherited from the previous chunk, which assumes timestamps are
            // non-decreasing in text order.
            previous.time_end = max_time(previous.time_end, piece.time_end);
            continue;
          }
        }
        merged.push(piece);
      }
    }
    merged
  };

  let mut chunks = Vec::new();
  for piece in pending {
    let text = &payload.text[piece.range];
    if text.trim().is_empty() {
      continue;
    }
    let index = chunks.len() as u32;
    chunks.push(RawChunk {
      index,
      text: text.to_string(),
      content_hash: chunk_hash(text),
      time_start: piece.time_start,
      time_end: piece.time_end,
    });
  }
  chunks
}

/// Hashes a chunk's text for cross-source dedup (`chunk.content_hash`).
///
/// Whitespace is collapsed first (`split_whitespace` + single-space join), so
/// formatting-only differences such as `"a  b\n"` and `"a b"` produce the same
/// key. This is safe because chunks are byte slices of an already-normalized
/// payload: the hash covers content, not layout.
///
/// Returns lowercase SHA-256 hex (64 characters). Callers must not parse it.
pub fn chunk_hash(text: &str) -> String {
  let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
  let digest = Sha256::digest(normalized.as_bytes());
  format!("{digest:x}")
}

/// A line range plus the turn metadata needed for grouping and time bounds.
///
/// [`TurnSpan`] covers only the turn text; a unit extends it to whole lines so a
/// chunk boundary can fall between turns instead of inside a `[ts] speaker: `
/// prefix.
#[derive(Debug, Clone, PartialEq)]
struct Unit {
  range: Range<usize>,
  at: Option<Timestamp>,
  session: Option<u32>,
}

/// A chunk range with its time bounds, before ids and hashes are assigned.
///
/// Kept private: `index` and `content_hash` are assigned only after
/// whitespace-only slices are dropped, so chunk indices stay dense.
#[derive(Debug, Clone)]
struct PendingChunk {
  range: Range<usize>,
  time_start: Option<Timestamp>,
  time_end: Option<Timestamp>,
}

/// Extends each span to whole lines and drops malformed spans.
///
/// For each span:
///
/// - `line_start` is the byte after the previous `\n` (or 0);
/// - `line_end` is the byte after the next `\n` (or the end of `text`);
/// - a multi-line turn therefore owns every line it touches;
/// - `cursor` keeps units ordered and non-overlapping even if spans overlap;
/// - spans outside `text` or with `start > end` are skipped instead of panicking.
///
/// Empty ranges are dropped; the result preserves text order.
fn build_units(text: &str, spans: &[TurnSpan]) -> Vec<Unit> {
  let mut units = Vec::with_capacity(spans.len());
  let mut cursor = 0;

  for span in spans {
    if span.byte_start > span.byte_end || span.byte_end > text.len() {
      continue;
    }
    let line_start = text[..span.byte_start].rfind('\n').map_or(0, |i| i + 1);
    let start = line_start.max(cursor);
    let line_end = text[span.byte_end..]
      .find('\n')
      .map_or(text.len(), |i| span.byte_end + i + 1);
    let end = line_end.max(start);
    if end > start {
      units.push(Unit {
        range: start..end,
        at: span.at,
        session: span.session,
      });
      cursor = end;
    }
  }

  units
}

/// Partitions ordered units into sessions.
///
/// A new session starts when the session id changes, or — with equal ids and
/// both timestamps present — when the gap exceeds `gap`. Groups are returned in
/// text order; an empty input yields no groups.
fn group_sessions(units: Vec<Unit>, gap: Duration) -> Vec<Vec<Unit>> {
  let mut groups: Vec<Vec<Unit>> = Vec::new();

  for unit in units {
    let starts_new = groups
      .last()
      .and_then(|group| group.last())
      .is_none_or(|previous| session_boundary(previous, &unit, gap));
    if starts_new {
      groups.push(Vec::new());
    }
    groups
      .last_mut()
      .expect("a group was just ensured")
      .push(unit);
  }

  groups
}

/// Returns `true` when `next` begins a new session.
///
/// The session id wins over time: a change always splits. With equal ids, only a
/// strictly larger gap splits; missing or out-of-order timestamps never do.
fn session_boundary(previous: &Unit, next: &Unit, gap: Duration) -> bool {
  if previous.session != next.session {
    return true;
  }
  match (previous.at, next.at) {
    (Some(earlier), Some(later)) => later.signed_duration_since(earlier) > gap,
    _ => false,
  }
}

/// Packs one session's units into chunks by greedy accumulation.
///
/// Units are appended while the estimate stays within `max_tokens`. When the
/// next unit would overflow, the current range is flushed and the unit starts a
/// new chunk — after splitting itself first if it alone exceeds the bound.
/// Resulting ranges are converted to [`PendingChunk`]s carrying the min/max turn
/// time they cover.
fn chunk_group(
  text: &str,
  units: &[Unit],
  max_tokens: usize,
  estimator: &dyn TokenEstimator,
) -> Vec<PendingChunk> {
  let mut ranges: Vec<Range<usize>> = Vec::new();
  let mut current: Option<Range<usize>> = None;

  for unit in units {
    match current.take() {
      None => current = place_unit(text, unit, &mut ranges, max_tokens, estimator),
      Some(range) => {
        let candidate = range.start..unit.range.end;
        if estimator.estimate(&text[candidate.clone()]) <= max_tokens {
          current = Some(candidate);
        } else {
          ranges.push(range);
          current = place_unit(text, unit, &mut ranges, max_tokens, estimator);
        }
      }
    }
  }
  if let Some(range) = current {
    ranges.push(range);
  }

  ranges
    .into_iter()
    .map(|range| {
      let (time_start, time_end) = time_range(units, &range);
      PendingChunk {
        range,
        time_start,
        time_end,
      }
    })
    .collect()
}

/// Starts a chunk at `unit`, splitting the unit itself when it exceeds the bound.
///
/// Returns the tail range to continue with: the whole unit when it fits, or the
/// last piece of an oversized unit after the earlier pieces were appended to
/// `out`. `None` means the unit produced no non-blank piece.
fn place_unit(
  text: &str,
  unit: &Unit,
  out: &mut Vec<Range<usize>>,
  max_tokens: usize,
  estimator: &dyn TokenEstimator,
) -> Option<Range<usize>> {
  if estimator.estimate(&text[unit.range.clone()]) <= max_tokens {
    return Some(unit.range.clone());
  }

  let mut pieces = split_range(text, unit.range.clone(), max_tokens, estimator);
  let last = pieces.pop();
  out.append(&mut pieces);
  last
}

/// Cuts `range` into pieces of at most `max_tokens` estimated tokens.
///
/// A binary search over char boundaries picks the longest prefix that fits. If
/// even one character does not fit, one character is still emitted, guaranteeing
/// progress (otherwise a tiny `max_tokens` would loop forever). Whitespace-only
/// pieces are skipped, so the returned ranges may cover fewer bytes than
/// `range`; every piece is a valid UTF-8 slice.
fn split_range(
  text: &str,
  range: Range<usize>,
  max_tokens: usize,
  estimator: &dyn TokenEstimator,
) -> Vec<Range<usize>> {
  let mut pieces = Vec::new();
  let mut start = range.start;

  while start < range.end {
    let remainder = start..range.end;
    if estimator.estimate(&text[remainder.clone()]) <= max_tokens {
      push_piece(&mut pieces, text, remainder);
      break;
    }

    // Candidate ends: the end of every char in the remainder, plus range.end.
    let mut ends: Vec<usize> = text[start..range.end]
      .char_indices()
      .skip(1)
      .map(|(offset, _)| start + offset)
      .collect();
    ends.push(range.end);

    // Largest prefix that still fits; index 0 always "fits" to guarantee progress.
    let mut low = 0;
    let mut high = ends.len();
    while low < high {
      let mid = (low + high) / 2;
      if estimator.estimate(&text[start..ends[mid]]) <= max_tokens {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    let chosen = if low == 0 { ends[0] } else { ends[low - 1] };

    push_piece(&mut pieces, text, start..chosen);
    start = chosen;
  }

  pieces
}

/// Pushes `range` unless it is empty or whitespace-only.
///
/// Dropping blank slices is what keeps chunks non-empty. The lossless promise
/// therefore excludes whitespace that carries no content.
fn push_piece(out: &mut Vec<Range<usize>>, text: &str, range: Range<usize>) {
  if range.start < range.end && !text[range.clone()].trim().is_empty() {
    out.push(range);
  }
}

/// Window-splits a document (a payload without turn metadata).
///
/// Documents carry no timestamps, so every chunk gets `None` time bounds.
fn document_chunks(
  text: &str,
  max_tokens: usize,
  estimator: &dyn TokenEstimator,
) -> Vec<PendingChunk> {
  split_range(text, 0..text.len(), max_tokens, estimator)
    .into_iter()
    .map(|range| PendingChunk {
      range,
      time_start: None,
      time_end: None,
    })
    .collect()
}

/// Returns the min/max turn time intersecting `range`.
///
/// Turns without a timestamp are ignored; if no timed turn overlaps, the result
/// is `(None, None)`.
fn time_range(units: &[Unit], range: &Range<usize>) -> (Option<Timestamp>, Option<Timestamp>) {
  let mut start: Option<Timestamp> = None;
  let mut end: Option<Timestamp> = None;

  for unit in units {
    if unit.range.start < range.end
      && unit.range.end > range.start
      && let Some(at) = unit.at
    {
      start = Some(start.map_or(at, |current| current.min(at)));
      end = Some(end.map_or(at, |current| current.max(at)));
    }
  }

  (start, end)
}

/// Takes the later of two optional timestamps; used when merging chunks.
fn max_time(a: Option<Timestamp>, b: Option<Timestamp>) -> Option<Timestamp> {
  match (a, b) {
    (Some(a), Some(b)) => Some(a.max(b)),
    (a, b) => a.or(b),
  }
}

#[cfg(test)]
mod tests {
  use chrono::TimeZone;

  use super::*;
  use crate::retain::normalize::{RawPayload, RawTurn, normalize};
  use crate::tokens::CharHeuristic;

  fn estimator() -> CharHeuristic {
    CharHeuristic::new(1.0) // 1 token per char keeps sizes predictable
  }

  fn config(max_tokens: usize) -> ChunkConfig {
    ChunkConfig {
      session_gap: Duration::minutes(30),
      max_tokens,
      min_tokens: 1,
    }
  }

  fn at(hour: u32, minute: u32) -> Timestamp {
    chrono::Utc
      .with_ymd_and_hms(2024, 6, 9, hour, minute, 0)
      .single()
      .expect("valid instant")
  }

  fn turn(text: &str, at: Option<Timestamp>, session: Option<u32>) -> RawTurn {
    RawTurn {
      speaker: "u".into(),
      text: text.into(),
      at,
      session,
    }
  }

  fn transcript(turns: Vec<RawTurn>) -> NormalizedPayload {
    normalize(RawPayload::Transcript { title: None, turns })
  }

  #[test]
  fn session_ids_group_turns_into_separate_chunks() {
    let payload = transcript(vec![
      turn("first", Some(at(12, 0)), Some(1)),
      turn("second", Some(at(12, 1)), Some(1)),
      turn("third", Some(at(13, 0)), Some(2)),
    ]);

    let chunks = chunk(&payload, &config(1_000), &estimator());

    assert_eq!(chunks.len(), 2);
    assert!(chunks[0].text.contains("first"));
    assert!(chunks[0].text.contains("second"));
    assert!(!chunks[0].text.contains("third"));
    assert_eq!(chunks[0].time_start, Some(at(12, 0)));
    assert_eq!(chunks[0].time_end, Some(at(12, 1)));
    assert!(chunks[1].text.contains("third"));
    assert_eq!(chunks[1].time_start, Some(at(13, 0)));
    assert_eq!(chunks[1].time_end, Some(at(13, 0)));
  }

  #[test]
  fn timestamp_gap_splits_when_session_ids_are_absent() {
    let payload = transcript(vec![
      turn("before", Some(at(12, 0)), None),
      turn("after", Some(at(14, 0)), None),
    ]);

    let chunks = chunk(&payload, &config(1_000), &estimator());

    assert_eq!(chunks.len(), 2);
    assert!(chunks[0].text.contains("before"));
    assert!(chunks[1].text.contains("after"));
  }

  #[test]
  fn small_gaps_and_missing_times_stay_in_one_session() {
    let payload = transcript(vec![
      turn("one", Some(at(12, 0)), None),
      turn("two", Some(at(12, 10)), None),
      turn("three", None, None),
    ]);

    let chunks = chunk(&payload, &config(1_000), &estimator());

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].time_start, Some(at(12, 0)));
    assert_eq!(chunks[0].time_end, Some(at(12, 10)));
  }

  #[test]
  fn oversized_session_splits_at_turn_boundaries() {
    // Each line is "u: " + 40 chars + "\n" = 44 chars = 44 tokens.
    let payload = transcript(vec![
      turn(&"a".repeat(40), None, Some(1)),
      turn(&"b".repeat(40), None, Some(1)),
      turn(&"c".repeat(40), None, Some(1)),
    ]);

    let chunks = chunk(&payload, &config(100), &estimator());

    assert_eq!(
      chunks.len(),
      2,
      "2 lines fit into 100 tokens, the third does not"
    );
    assert!(chunks[0].text.contains(&"a".repeat(40)));
    assert!(chunks[0].text.contains(&"b".repeat(40)));
    assert!(chunks[1].text.contains(&"c".repeat(40)));
  }

  #[test]
  fn oversized_turn_is_split_without_losing_text() {
    let body = "x".repeat(1_000);
    let payload = transcript(vec![turn(&body, Some(at(12, 0)), Some(1))]);

    let chunks = chunk(&payload, &config(100), &estimator());

    assert!(chunks.len() > 1);
    assert!(chunks.iter().all(|chunk| !chunk.text.trim().is_empty()));
    assert!(
      chunks
        .iter()
        .all(|chunk| estimator().estimate(&chunk.text) <= 100)
    );
    let reassembled: String = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    assert_eq!(reassembled, payload.text);
    assert!(
      chunks
        .iter()
        .all(|chunk| chunk.time_start == Some(at(12, 0)))
    );
  }

  #[test]
  fn document_windows_cover_the_whole_text() {
    let text = "y".repeat(250);
    let payload = normalize(RawPayload::Document {
      title: None,
      text: text.clone(),
    });

    let chunks = chunk(&payload, &config(100), &estimator());

    assert_eq!(chunks.len(), 3);
    let reassembled: String = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    assert_eq!(reassembled, text);
    assert_eq!(
      chunks.iter().map(|chunk| chunk.index).collect::<Vec<_>>(),
      vec![0, 1, 2]
    );
    assert!(chunks.iter().all(|chunk| chunk.time_start.is_none()));
  }

  #[test]
  fn cjk_documents_split_on_char_boundaries() {
    let text = "你".repeat(120);
    let payload = normalize(RawPayload::Document {
      title: None,
      text: text.clone(),
    });

    let chunks = chunk(&payload, &config(50), &estimator());

    let reassembled: String = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    assert_eq!(reassembled, text);
    assert!(chunks.iter().all(|chunk| !chunk.text.is_empty()));
  }

  #[test]
  fn small_chunks_merge_into_the_previous_chunk() {
    let payload = transcript(vec![
      turn(&"a".repeat(60), None, Some(1)),
      turn(&"b".repeat(20), None, Some(2)),
    ]);
    let mut cfg = config(200);
    cfg.min_tokens = 50;

    let chunks = chunk(&payload, &cfg, &estimator());

    assert_eq!(chunks.len(), 1, "the small second session is merged");
    assert!(chunks[0].text.contains(&"a".repeat(60)));
    assert!(chunks[0].text.contains(&"b".repeat(20)));
  }

  #[test]
  fn chunking_is_deterministic() {
    let payload = transcript(vec![
      turn("first", Some(at(12, 0)), Some(1)),
      turn("second", Some(at(12, 1)), Some(1)),
    ]);

    let first = chunk(&payload, &config(1_000), &estimator());
    let second = chunk(&payload, &config(1_000), &estimator());

    assert_eq!(first, second);
  }

  #[test]
  fn empty_payloads_produce_no_chunks() {
    let empty = normalize(RawPayload::Document {
      title: None,
      text: String::new(),
    });
    let whitespace = normalize(RawPayload::Document {
      title: None,
      text: " \n\t ".into(),
    });
    let no_turns = empty_transcript();

    assert!(chunk(&empty, &config(100), &estimator()).is_empty());
    assert!(chunk(&whitespace, &config(100), &estimator()).is_empty());
    assert!(chunk(&no_turns, &config(100), &estimator()).is_empty());
  }

  fn empty_transcript() -> NormalizedPayload {
    normalize(RawPayload::Transcript {
      title: None,
      turns: Vec::new(),
    })
  }

  #[test]
  fn hash_ignores_whitespace_differences() {
    assert_eq!(
      chunk_hash("hello  world\n next"),
      chunk_hash("hello world next")
    );
    assert_ne!(chunk_hash("hello world"), chunk_hash("hello there"));
    assert_eq!(chunk_hash("hello").len(), 64);
  }

  #[test]
  fn build_units_skips_malformed_spans() {
    let text = "a: one\nb: two\n";
    let spans = vec![
      TurnSpan {
        byte_start: 3,
        byte_end: 6,
        at: None,
        session: None,
      },
      // Out of range: ignored instead of panicking.
      TurnSpan {
        byte_start: 50,
        byte_end: 60,
        at: None,
        session: None,
      },
      // Reversed: ignored.
      TurnSpan {
        byte_start: 5,
        byte_end: 4,
        at: None,
        session: None,
      },
      TurnSpan {
        byte_start: 10,
        byte_end: 13,
        at: None,
        session: None,
      },
    ];

    let units = build_units(text, &spans);

    assert_eq!(units.len(), 2);
    assert_eq!(units[0].range, 0..7);
    assert_eq!(units[1].range, 7..14);
  }

  /// An estimator that always reports "too large", to force the one-char
  /// fallback in [`split_range`].
  struct AlwaysOver;

  impl TokenEstimator for AlwaysOver {
    fn estimate(&self, text: &str) -> usize {
      if text.is_empty() { 0 } else { usize::MAX }
    }
  }

  #[test]
  fn split_range_always_advances_by_one_char() {
    let text = "abc";

    let pieces = split_range(text, 0..text.len(), 1, &AlwaysOver);

    assert_eq!(pieces.len(), 3);
    assert_eq!(
      pieces
        .iter()
        .map(|piece| &text[piece.clone()])
        .collect::<String>(),
      "abc"
    );
    assert!(pieces.iter().all(|piece| piece.end - piece.start == 1));
  }

  #[test]
  fn split_range_never_splits_a_multibyte_char() {
    let text = "你好";

    let pieces = split_range(text, 0..text.len(), 1, &AlwaysOver);

    assert_eq!(pieces, vec![0..3, 3..6]);
    assert_eq!(&text[pieces[0].clone()], "你");
    assert_eq!(&text[pieces[1].clone()], "好");
  }

  #[test]
  fn whitespace_only_slices_are_never_emitted() {
    let text = format!("a{}b", " ".repeat(60));
    let payload = normalize(RawPayload::Document {
      title: None,
      text: text.clone(),
    });

    let chunks = chunk(&payload, &config(10), &estimator());

    assert!(chunks.iter().all(|chunk| !chunk.text.trim().is_empty()));
    let content: String = chunks.iter().map(|chunk| chunk.text.trim()).collect();
    assert_eq!(content, "ab");
    assert!(chunks.len() < text.len(), "blank windows were dropped");
  }

  #[test]
  fn max_time_keeps_the_later_bound() {
    let earlier = at(12, 0);
    let later = at(13, 0);

    assert_eq!(max_time(Some(earlier), Some(later)), Some(later));
    assert_eq!(max_time(Some(later), Some(earlier)), Some(later));
    assert_eq!(max_time(Some(earlier), None), Some(earlier));
    assert_eq!(max_time(None, Some(later)), Some(later));
    assert_eq!(max_time(None, None), None);
  }

  fn unit(at: Option<Timestamp>, session: Option<u32>) -> Unit {
    Unit {
      range: 0..1,
      at,
      session,
    }
  }

  #[test]
  fn session_boundary_ignores_out_of_order_and_equal_gaps() {
    let gap = Duration::minutes(30);

    // Out-of-order timestamps never split.
    assert!(!session_boundary(
      &unit(Some(at(13, 0)), None),
      &unit(Some(at(12, 0)), None),
      gap
    ));
    // The comparison is strict: a gap equal to the threshold does not split.
    assert!(!session_boundary(
      &unit(Some(at(12, 0)), None),
      &unit(Some(at(12, 30)), None),
      gap
    ));
    assert!(session_boundary(
      &unit(Some(at(12, 0)), None),
      &unit(Some(at(12, 31)), None),
      gap
    ));
    // Session ids win over time.
    assert!(session_boundary(
      &unit(None, Some(1)),
      &unit(None, Some(2)),
      gap
    ));
    // Same id, no timestamps: same session.
    assert!(!session_boundary(&unit(None, None), &unit(None, None), gap));
  }

  fn ranged_unit(index: usize, at: Option<Timestamp>, session: Option<u32>) -> Unit {
    Unit {
      range: index..index + 1,
      at,
      session,
    }
  }

  #[test]
  fn group_sessions_handles_empty_input() {
    assert!(group_sessions(Vec::new(), Duration::minutes(30)).is_empty());
  }

  #[test]
  fn group_sessions_splits_on_id_changes_and_large_gaps() {
    let gap = Duration::minutes(30);
    let units = vec![
      ranged_unit(0, Some(at(12, 0)), Some(1)),
      ranged_unit(1, Some(at(12, 10)), Some(1)),
      ranged_unit(2, Some(at(12, 20)), Some(2)),
      // Same id as the previous unit, but a 100-minute gap exceeds the threshold.
      ranged_unit(3, Some(at(14, 0)), Some(2)),
    ];

    let groups = group_sessions(units, gap);

    assert_eq!(groups.len(), 3);
    assert_eq!(groups[0].len(), 2);
    assert_eq!(groups[1].len(), 1);
    assert_eq!(groups[2].len(), 1);

    // Every group carries the session ids expected for that run.
    let sessions: Vec<Vec<Option<u32>>> = groups
      .iter()
      .map(|group| group.iter().map(|unit| unit.session).collect())
      .collect();
    assert_eq!(
      sessions,
      vec![vec![Some(1), Some(1)], vec![Some(2)], vec![Some(2)]]
    );
    // The last group is a new group despite the same session id: the gap splits it.
    assert_eq!(groups[1][0].at, Some(at(12, 20)));
    assert_eq!(groups[2][0].at, Some(at(14, 0)));

    // Order is preserved, every unit appears exactly once, ranges stay intact.
    let starts: Vec<usize> = groups
      .iter()
      .flat_map(|group| group.iter().map(|unit| unit.range.start))
      .collect();
    assert_eq!(starts, vec![0, 1, 2, 3]);
  }

  #[test]
  fn group_sessions_prefers_ids_over_time() {
    let units = vec![
      ranged_unit(0, None, Some(1)),
      ranged_unit(1, None, Some(1)),
      ranged_unit(2, None, Some(3)),
    ];

    let groups = group_sessions(units, Duration::minutes(30));

    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].len(), 2);
    assert_eq!(groups[1].len(), 1);
    assert_eq!(groups[0][0].range.start, 0);
    assert_eq!(groups[0][1].range.start, 1);
    assert_eq!(groups[1][0].range.start, 2);
  }

  #[test]
  fn chunk_group_flushes_when_the_next_unit_overflows() {
    let text = "a\nb\nc\n"; // units: 0..2, 2..4, 4..6
    let units = vec![
      Unit {
        range: 0..2,
        at: Some(at(12, 0)),
        session: Some(1),
      },
      Unit {
        range: 2..4,
        at: Some(at(12, 1)),
        session: Some(1),
      },
      Unit {
        range: 4..6,
        at: Some(at(12, 2)),
        session: Some(1),
      },
    ];

    let chunks = chunk_group(text, &units, 3, &estimator());

    assert_eq!(chunks.len(), 3, "one unit per chunk at 3 tokens");
    assert_eq!(
      chunks
        .iter()
        .map(|chunk| chunk.range.clone())
        .collect::<Vec<_>>(),
      vec![0..2, 2..4, 4..6]
    );
    assert_eq!(chunks[0].time_start, Some(at(12, 0)));
    assert_eq!(chunks[0].time_end, Some(at(12, 0)));
    assert_eq!(chunks[2].time_start, Some(at(12, 2)));
    assert_eq!(chunks[2].time_end, Some(at(12, 2)));
  }

  #[test]
  fn chunk_group_merges_units_that_fit_and_keeps_time_bounds() {
    let text = "aa\nbb\ncc\n"; // units: 0..3, 3..6, 6..9
    let units = vec![
      Unit {
        range: 0..3,
        at: Some(at(12, 0)),
        session: None,
      },
      Unit {
        range: 3..6,
        at: Some(at(12, 1)),
        session: None,
      },
      Unit {
        range: 6..9,
        at: Some(at(12, 2)),
        session: None,
      },
    ];

    let chunks = chunk_group(text, &units, 6, &estimator());

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].range, 0..6, "two units fit exactly");
    assert_eq!(chunks[0].time_start, Some(at(12, 0)));
    assert_eq!(chunks[0].time_end, Some(at(12, 1)));
    assert_eq!(chunks[1].range, 6..9);
    assert_eq!(chunks[1].time_start, Some(at(12, 2)));
    assert_eq!(chunks[1].time_end, Some(at(12, 2)));
  }

  #[test]
  fn chunk_group_splits_an_oversized_unit_and_keeps_time_bounds() {
    let text = format!("ab\n{}", "x".repeat(25)); // 28 chars; units 0..3 and 3..28
    let units = vec![
      Unit {
        range: 0..3,
        at: Some(at(12, 0)),
        session: Some(1),
      },
      Unit {
        range: 3..text.len(),
        at: Some(at(13, 0)),
        session: Some(1),
      },
    ];

    let chunks = chunk_group(&text, &units, 10, &estimator());

    assert_eq!(
      chunks.len(),
      4,
      "small unit + three pieces of the oversized unit"
    );
    assert_eq!(
      chunks
        .iter()
        .map(|chunk| chunk.range.clone())
        .collect::<Vec<_>>(),
      vec![0..3, 3..13, 13..23, 23..28]
    );
    assert!(
      chunks
        .iter()
        .all(|chunk| estimator().estimate(&text[chunk.range.clone()]) <= 10)
    );
    // The first chunk belongs to the timed small unit, the rest inherit the
    // oversized unit's timestamp.
    assert_eq!(chunks[0].time_start, Some(at(12, 0)));
    assert_eq!(chunks[0].time_end, Some(at(12, 0)));
    for chunk in &chunks[1..] {
      assert_eq!(chunk.time_start, Some(at(13, 0)));
      assert_eq!(chunk.time_end, Some(at(13, 0)));
    }

    let reassembled: String = chunks
      .iter()
      .map(|chunk| &text[chunk.range.clone()])
      .collect();
    assert_eq!(reassembled, text);
  }

  #[test]
  fn chunk_group_handles_no_units_and_untimed_units() {
    assert!(chunk_group("", &[], 10, &estimator()).is_empty());

    let text = "a\nb\n";
    let units = vec![
      Unit {
        range: 0..2,
        at: None,
        session: None,
      },
      Unit {
        range: 2..4,
        at: None,
        session: None,
      },
    ];

    let chunks = chunk_group(text, &units, 10, &estimator());

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].range, 0..4);
    assert_eq!(chunks[0].time_start, None);
    assert_eq!(chunks[0].time_end, None);
  }

  #[test]
  fn place_unit_returns_the_tail_of_an_oversized_unit() {
    let text = format!("u: {}", "x".repeat(100));
    let unit = Unit {
      range: 0..text.len(),
      at: Some(at(12, 0)),
      session: Some(1),
    };
    let mut earlier = Vec::new();

    let tail = place_unit(&text, &unit, &mut earlier, 10, &estimator()).expect("tail piece");

    assert!(!earlier.is_empty());
    let mut all = earlier;
    all.push(tail);
    assert_eq!(&text[all[0].start..all.last().expect("last").end], text);
    assert!(all.windows(2).all(|pair| pair[0].end == pair[1].start));
    assert!(
      all
        .iter()
        .all(|range| estimator().estimate(&text[range.clone()]) <= 10)
    );
  }

  #[test]
  fn place_unit_returns_none_for_a_blank_unit() {
    let text = "   ";
    let unit = Unit {
      range: 0..text.len(),
      at: None,
      session: None,
    };
    let mut out = Vec::new();

    assert_eq!(place_unit(text, &unit, &mut out, 1, &estimator()), None);
    assert!(out.is_empty());
  }
}
