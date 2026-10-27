//! Payload normalization: caller input → canonical text + a turn view.
//!
//! The canonical text is the durable artifact (`source.payload_text`); the turn
//! spans are an in-memory view used by chunking. Spans are deliberately **not**
//! persisted: a crash between tx0 and tx1 falls back to window chunking (see
//! `design/memory-schema.md` §5.1), which is why retain acks only after tx1.

use chrono::SecondsFormat;

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
