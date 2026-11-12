//! Payload normalization: caller input → canonical text + a turn view.
//!
//! The canonical text is the durable artifact (`source.payload_text`); the turn
//! spans are an in-memory view used by chunking. Spans are deliberately **not**
//! persisted: a crash between tx0 and tx1 falls back to window chunking (see
//! `design/memory-schema.md` §5.1), which is why retain acks only after tx1.
//! [`recover_time_hints`] reparses the durable `[<ts>]` prefixes so that crash
//! recovery does not also lose chunk time bounds; session ids remain unrecoverable.

use chrono::{DateTime, SecondsFormat};

use crate::types::Timestamp;

/// One utterance supplied by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct RawTurn {
  /// Speaker label, e.g. `"user"` or `"assistant"`.
  pub speaker: String,
  /// Verbatim utterance text (multi-line is allowed).
  pub text: String,
  /// Wall-clock time of the utterance, when the caller provides one.
  pub at: Option<Timestamp>,
  /// Session grouping hint; preserved in spans for chunking alignment.
  pub session: Option<u32>,
}

/// The payload shapes accepted by retain.
#[derive(Debug, Clone, PartialEq)]
pub enum RawPayload {
  /// A conversation transcript with turn boundaries.
  Transcript {
    /// Optional human-readable title, stored on the source row.
    title: Option<String>,
    /// Turns in conversation order.
    turns: Vec<RawTurn>,
  },
  /// A plain document; its text is already canonical.
  Document {
    /// Optional human-readable title, stored on the source row.
    title: Option<String>,
    /// Verbatim document text.
    text: String,
  },
}

impl RawPayload {
  /// Returns the payload title, which retain stores on the source row.
  pub fn title(&self) -> Option<&str> {
    match self {
      Self::Transcript { title, .. } | Self::Document { title, .. } => title.as_deref(),
    }
  }
}

/// A turn text as a byte range inside [`NormalizedPayload::text`].
///
/// The range covers **only the turn text** — not the `[<ts>] <speaker>: ` prefix
/// and not the trailing newline — so `&text[byte_start..byte_end] == turn.text`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnSpan {
  /// Inclusive start byte offset.
  pub byte_start: usize,
  /// Exclusive end byte offset.
  pub byte_end: usize,
  /// Wall-clock time of the utterance, when supplied.
  pub at: Option<Timestamp>,
  /// Session grouping hint, when supplied.
  pub session: Option<u32>,
}

/// A timestamp recovered from a canonical transcript line.
///
/// `start..end` is the byte range covered by the timestamped turn: it begins at
/// the timestamped line and ends at the next recognized turn start (timed or
/// untimed), or the text end. This stays coarser than [`TurnSpan`]: after a
/// crash the session ids are gone, and multi-line text can confuse boundary
/// detection, but the range is enough to attach time bounds to window chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeHint {
  /// Inclusive start byte offset (the timestamped line).
  pub start: usize,
  /// Exclusive end byte offset (the next timestamped line, or text end).
  pub end: usize,
  /// Timestamp carried by the line.
  pub at: Timestamp,
}

/// Canonical text plus its turn view.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedPayload {
  /// Canonical full text; becomes `source.payload_text`.
  pub text: String,
  /// Turn text ranges; empty for documents.
  pub spans: Vec<TurnSpan>,
}

/// Converts a payload into canonical text and spans.
///
/// Transcript lines are rendered as `"[<ts>] <speaker>: <text>\n"`; the
/// `"[<ts>] "` prefix is omitted when the turn has no timestamp, so no time is
/// ever fabricated. Documents pass through byte-for-byte.
pub fn normalize(payload: RawPayload) -> NormalizedPayload {
  match payload {
    RawPayload::Document { text, .. } => NormalizedPayload {
      text,
      spans: Vec::new(),
    },
    RawPayload::Transcript { turns, .. } => {
      let mut text = String::new();
      let mut spans = Vec::with_capacity(turns.len());

      for turn in turns {
        if let Some(at) = turn.at {
          text.push('[');
          text.push_str(&at.to_rfc3339_opts(SecondsFormat::Millis, true));
          text.push_str("] ");
        }
        text.push_str(&turn.speaker);
        text.push_str(": ");

        let byte_start = text.len();
        text.push_str(&turn.text);
        let byte_end = text.len();
        text.push('\n');

        spans.push(TurnSpan {
          byte_start,
          byte_end,
          at: turn.at,
          session: turn.session,
        });
      }

      NormalizedPayload { text, spans }
    }
  }
}

/// Recovers time hints from canonical transcript text after a crash.
///
/// A line starting with `[<rfc3339>] <speaker>: ` is a timed turn; a line
/// starting with `<speaker>: ` is an untimed turn. The untimed boundary matters:
/// it ends the preceding hint, so a timed turn's time does not bleed into a
/// following untimed turn. Malformed timestamp prefixes are ignored.
///
/// The result is empty for documents and untimed transcripts, so callers keep
/// the ordinary window fallback. Because canonical text does not escape
/// newlines, a multi-line turn whose continuation happens to look like
/// `speaker: ...` can end a hint early; recovery is intentionally conservative
/// about adding times rather than inventing them.
pub fn recover_time_hints(text: &str) -> Vec<TimeHint> {
  let mut starts: Vec<(usize, Option<Timestamp>)> = Vec::new();
  let mut line_start = 0;

  while line_start < text.len() {
    let line_end = text[line_start..]
      .find('\n')
      .map_or(text.len(), |offset| line_start + offset);
    if let Some(at) = parse_turn_start(&text[line_start..line_end]) {
      starts.push((line_start, at));
    }
    if line_end == text.len() {
      break;
    }
    line_start = line_end + 1;
  }

  let mut hints = Vec::new();
  for (index, (start, at)) in starts.iter().enumerate() {
    let Some(at) = at else {
      continue;
    };
    let end = starts.get(index + 1).map_or(text.len(), |(next, _)| *next);
    hints.push(TimeHint {
      start: *start,
      end,
      at: *at,
    });
  }
  hints
}

/// Parses one canonical line as `Some(timestamp)` (timed turn),
/// `Some(None)` (untimed turn), or `None` (not a turn start).
fn parse_turn_start(line: &str) -> Option<Option<Timestamp>> {
  if line.starts_with('[') {
    let rest = line.strip_prefix('[')?;
    let close = rest.find("] ")?;
    let stamp = rest.get(..close)?;
    let at = DateTime::parse_from_rfc3339(stamp)
      .ok()?
      .with_timezone(&chrono::Utc);
    let after = rest.get(close + 2..)?;
    if !after.contains(": ") {
      return None;
    }
    return Some(Some(at));
  }

  let colon = line.find(": ")?;
  let speaker = line.get(..colon)?;
  if speaker.trim().is_empty() || speaker.chars().count() > 64 {
    return None;
  }
  Some(None)
}

#[cfg(test)]
mod tests {
  use chrono::TimeZone;

  use super::*;

  fn at(hour: u32, minute: u32) -> Timestamp {
    chrono::Utc
      .with_ymd_and_hms(2024, 6, 9, hour, minute, 0)
      .single()
      .expect("valid instant")
  }

  fn turn(speaker: &str, text: &str) -> RawTurn {
    RawTurn {
      speaker: speaker.into(),
      text: text.into(),
      at: None,
      session: None,
    }
  }

  #[test]
  fn document_passes_through_byte_for_byte() {
    let payload = RawPayload::Document {
      title: Some("notes".into()),
      text: "first line\n第二行\n".into(),
    };

    assert_eq!(payload.title(), Some("notes"));
    let normalized = normalize(payload);

    assert_eq!(normalized.text, "first line\n第二行\n");
    assert!(normalized.spans.is_empty());
  }

  #[test]
  fn transcript_renders_timestamped_and_plain_lines() {
    let payload = RawPayload::Transcript {
      title: None,
      turns: vec![
        RawTurn {
          speaker: "user".into(),
          text: "Hi".into(),
          at: Some(at(12, 0)),
          session: Some(1),
        },
        turn("assistant", "Hello"),
      ],
    };

    let normalized = normalize(payload);

    assert_eq!(
      normalized.text,
      "[2024-06-09T12:00:00.000Z] user: Hi\nassistant: Hello\n"
    );
    assert_eq!(normalized.spans.len(), 2);

    let first = normalized.spans[0];
    assert_eq!(&normalized.text[first.byte_start..first.byte_end], "Hi");
    assert_eq!(first.at, Some(at(12, 0)));
    assert_eq!(first.session, Some(1));

    let second = normalized.spans[1];
    assert_eq!(
      &normalized.text[second.byte_start..second.byte_end],
      "Hello"
    );
    assert_eq!(second.at, None);
    assert_eq!(second.session, None);

    // No timestamp means no fabricated prefix.
    assert!(normalized.text[second.byte_start..].starts_with("Hello\n"));
    let line_start = normalized.text.rfind("assistant:").expect("speaker");
    assert_eq!(&normalized.text[line_start..line_start + 10], "assistant:");
  }

  #[test]
  fn empty_transcript_normalizes_to_empty_text() {
    let normalized = normalize(RawPayload::Transcript {
      title: Some("empty".into()),
      turns: Vec::new(),
    });

    assert!(normalized.text.is_empty());
    assert!(normalized.spans.is_empty());
  }

  #[test]
  fn cjk_offsets_are_bytes_not_chars() {
    let text = "你好，Emily";
    let normalized = normalize(RawPayload::Transcript {
      title: None,
      turns: vec![turn("用户", text)],
    });

    let span = normalized.spans[0];
    assert_eq!(&normalized.text[span.byte_start..span.byte_end], text);
    assert_eq!(span.byte_end - span.byte_start, text.len());
    assert!(span.byte_start > 0, "the speaker prefix is excluded");
  }

  #[test]
  fn multi_line_turn_text_stays_invertible() {
    let text = "first line\nsecond line";
    let normalized = normalize(RawPayload::Transcript {
      title: None,
      turns: vec![turn("user", text)],
    });

    let span = normalized.spans[0];
    assert_eq!(&normalized.text[span.byte_start..span.byte_end], text);
    assert!(normalized.text.ends_with("\n"));
  }

  #[test]
  fn very_long_turn_is_handled_without_truncation() {
    let text = "a".repeat(100_000);
    let normalized = normalize(RawPayload::Transcript {
      title: None,
      turns: vec![RawTurn {
        speaker: "user".into(),
        text: text.clone(),
        at: None,
        session: None,
      }],
    });

    let span = normalized.spans[0];
    assert_eq!(span.byte_end - span.byte_start, text.len());
    assert_eq!(&normalized.text[span.byte_start..span.byte_end], text);
  }

  #[test]
  fn recovers_time_hints_from_canonical_text() {
    let payload = normalize(RawPayload::Transcript {
      title: None,
      turns: vec![
        RawTurn {
          speaker: "user".into(),
          text: "first".into(),
          at: Some(at(12, 0)),
          session: Some(1),
        },
        turn("assistant", "untimed"),
        RawTurn {
          speaker: "user".into(),
          text: "second".into(),
          at: Some(at(12, 1)),
          session: Some(1),
        },
      ],
    });

    let hints = recover_time_hints(&payload.text);
    assert_eq!(hints.len(), 2);
    assert_eq!(hints[0].at, at(12, 0));
    assert_eq!(hints[1].at, at(12, 1));
    assert_eq!(hints[0].start, 0);

    let untimed = payload
      .text
      .find("assistant: untimed")
      .expect("untimed turn");
    let second = payload
      .text
      .find("[2024-06-09T12:01:00.000Z]")
      .expect("second prefix");
    assert_eq!(
      hints[0].end, untimed,
      "the untimed turn ends the first timed hint"
    );
    assert_eq!(hints[1].start, second);
    assert_eq!(hints[1].end, payload.text.len());
  }

  #[test]
  fn recovery_ignores_untimed_and_malformed_lines() {
    let text = "[not-a-time] user: no\n\
                [2024-06-09T12:00:00.000Z] no speaker separator\n\
                [2024-06-09T12:00:00.000Z] user: yes\n";

    let hints = recover_time_hints(text);
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0].at, at(12, 0));
    assert_eq!(
      hints[0].start,
      text
        .find("[2024-06-09T12:00:00.000Z] user: yes")
        .expect("line")
    );
    assert_eq!(hints[0].end, text.len());
  }

  #[test]
  fn documents_without_timestamps_recover_no_hints() {
    assert!(recover_time_hints("").is_empty());
    assert!(recover_time_hints("plain document\nno timestamps\n").is_empty());
  }

  #[test]
  fn recovery_handles_cjk_line_offsets() {
    let text = "[2024-06-09T12:00:00.000Z] 用户: 你好\n\
                [2024-06-09T12:01:00.000Z] 用户: 再见\n";

    let hints = recover_time_hints(text);
    assert_eq!(hints.len(), 2);
    let second = text.find("[2024-06-09T12:01:00.000Z]").expect("second");
    assert_eq!(hints[0].start, 0);
    assert_eq!(hints[0].end, second);
    assert_eq!(hints[1].start, second);
  }

  #[test]
  fn title_accessor_covers_both_payload_shapes() {
    let transcript = RawPayload::Transcript {
      title: Some("chat".into()),
      turns: Vec::new(),
    };
    let document = RawPayload::Document {
      title: None,
      text: String::new(),
    };

    assert_eq!(transcript.title(), Some("chat"));
    assert_eq!(document.title(), None);
  }
}
