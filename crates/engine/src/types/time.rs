//! Time primitives shared by the domain model and the ports.

use serde::{Deserialize, Serialize};

/// A UTC timestamp. All wall-clock values in the system are UTC.
pub type Timestamp = chrono::DateTime<chrono::Utc>;

/// Converts a timestamp to UTC epoch milliseconds — the logical `TS`
/// representation shared by adapters that store integers.
pub fn to_epoch_millis(timestamp: Timestamp) -> i64 {
  timestamp.timestamp_millis()
}

/// Converts UTC epoch milliseconds back to a timestamp.
///
/// Returns `None` when `millis` is outside chrono's representable range.
pub fn from_epoch_millis(millis: i64) -> Option<Timestamp> {
  Timestamp::from_timestamp_millis(millis)
}

/// A closed time interval in UTC, used as the temporal-retrieval window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
  /// Inclusive start.
  pub start: Timestamp,
  /// Inclusive end.
  pub end: Timestamp,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn epoch_millis_round_trip() {
    let timestamp = from_epoch_millis(1_700_000_000_123).expect("valid timestamp");

    assert_eq!(to_epoch_millis(timestamp), 1_700_000_000_123);
  }

  #[test]
  fn negative_millis_are_before_the_epoch() {
    let timestamp = from_epoch_millis(-1).expect("valid timestamp");

    assert_eq!(to_epoch_millis(timestamp), -1);
    assert!(timestamp.timestamp() < 0);
  }

  #[test]
  fn out_of_range_millis_are_rejected() {
    assert!(from_epoch_millis(i64::MAX).is_none());
    assert!(from_epoch_millis(i64::MIN).is_none());
  }
}
