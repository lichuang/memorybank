//! SQLite-specific time encodings that are not domain conventions.
//!
//! Epoch-millisecond conversions live in `engine::types` because the logical
//! `TS` type defines millisecond precision; this module keeps only helpers that
//! exist purely for adapter bookkeeping.

use std::time::Duration;

/// Converts a lease duration to epoch milliseconds, saturating on overflow.
pub(crate) fn duration_millis(duration: Duration) -> i64 {
  i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn converts_and_saturates() {
    assert_eq!(duration_millis(Duration::ZERO), 0);
    assert_eq!(duration_millis(Duration::from_millis(500)), 500);
    assert_eq!(duration_millis(Duration::from_secs(2)), 2_000);
    assert_eq!(duration_millis(Duration::MAX), i64::MAX);
  }
}
