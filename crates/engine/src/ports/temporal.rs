//! Query-side temporal parsing port.

use crate::types::{TimeRange, Timestamp};

/// Maps natural-language time expressions ("yesterday", "last week", ISO dates)
/// to a concrete interval.
///
/// The parser is query-side only: extraction already normalizes times into the
/// five Ws, so there is no write-side date parsing.
pub trait TemporalParser: Send + Sync {
  /// Returns `None` when `query` contains no recognizable time expression.
  fn parse(&self, query: &str, now: Timestamp) -> Option<TimeRange>;
}

#[cfg(test)]
mod tests {
  use chrono::TimeZone;

  use super::*;

  struct StubParser;

  impl TemporalParser for StubParser {
    fn parse(&self, query: &str, now: Timestamp) -> Option<TimeRange> {
      query.contains("today").then_some(TimeRange {
        start: now,
        end: now,
      })
    }
  }

  #[test]
  fn dyn_parser_is_injectable() {
    let parser: Box<dyn TemporalParser> = Box::new(StubParser);
    let now = chrono::Utc.with_ymd_and_hms(2024, 6, 9, 12, 0, 0).unwrap();

    assert!(parser.parse("what happened today?", now).is_some());
    assert!(parser.parse("what happened?", now).is_none());
  }
}
