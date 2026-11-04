//! Occurrence intervals as validated values.
//!
//! The database stores τs/τe as two nullable columns, but the engine must never
//! materialize an invalid combination (a reversed interval, or an end without a
//! start). [`OccurrenceInterval::from_parts`] is the only conversion from raw
//! columns or model output, and [`ClosedInterval`] keeps its fields private so a
//! reversed closed interval cannot be constructed.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::time::Timestamp;

/// Error returned when raw occurrence bounds do not form a valid interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntervalError {
  /// The end precedes the start.
  Reversed {
    /// Supplied start.
    start: Timestamp,
    /// Supplied end.
    end: Timestamp,
  },
  /// An end without a start; `(-∞, end]` has no meaning for a memory.
  EndWithoutStart {
    /// Supplied end.
    end: Timestamp,
  },
}

impl fmt::Display for IntervalError {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Reversed { start, end } => {
        write!(
          formatter,
          "reversed occurrence interval: {start} is after {end}"
        )
      }
      Self::EndWithoutStart { end } => {
        write!(formatter, "occurrence end {end} has no start")
      }
    }
  }
}

impl std::error::Error for IntervalError {}

/// A closed occurrence interval (`start <= end`).
///
/// Fields are private on purpose: the only constructor is [`ClosedInterval::new`],
/// which rejects reversed bounds. Deserialization goes through the same check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ClosedIntervalWire")]
pub struct ClosedInterval {
  start: Timestamp,
  end: Timestamp,
}

/// Deserialization mirror; [`TryFrom`] validates it before a `ClosedInterval`
/// can exist.
#[derive(Deserialize)]
struct ClosedIntervalWire {
  start: Timestamp,
  end: Timestamp,
}

impl TryFrom<ClosedIntervalWire> for ClosedInterval {
  type Error = IntervalError;

  fn try_from(wire: ClosedIntervalWire) -> Result<Self, Self::Error> {
    Self::new(wire.start, wire.end)
  }
}

impl ClosedInterval {
  /// Creates a closed interval, rejecting `end < start`.
  pub fn new(start: Timestamp, end: Timestamp) -> Result<Self, IntervalError> {
    if end < start {
      return Err(IntervalError::Reversed { start, end });
    }

    Ok(Self { start, end })
  }

  /// Interval start τs.
  pub fn start(&self) -> Timestamp {
    self.start
  }

  /// Interval end τe.
  pub fn end(&self) -> Timestamp {
    self.end
  }
}

/// A fact's occurrence time: closed, open (still ongoing), or unknown.
///
/// This is the domain representation of `fact.occurred_start` / `occurred_end`.
/// It makes the invalid end-only and reversed combinations unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OccurrenceInterval {
  /// Both bounds are known and ordered.
  Closed(ClosedInterval),
  /// The event is still ongoing; only the start is known.
  Open {
    /// Start of the interval.
    start: Timestamp,
  },
  /// No temporal information.
  Unknown,
}

impl OccurrenceInterval {
  /// Converts raw nullable columns/model fields into a validated interval.
  ///
  /// - `(None, None)` → [`OccurrenceInterval::Unknown`]
  /// - `(Some, None)` → [`OccurrenceInterval::Open`]
  /// - `(Some, Some)` → [`OccurrenceInterval::Closed`] or
  ///   [`IntervalError::Reversed`]
  /// - `(None, Some)` → [`IntervalError::EndWithoutStart`]
  pub fn from_parts(
    start: Option<Timestamp>,
    end: Option<Timestamp>,
  ) -> Result<Self, IntervalError> {
    match (start, end) {
      (None, None) => Ok(Self::Unknown),
      (Some(start), None) => Ok(Self::Open { start }),
      (Some(start), Some(end)) => ClosedInterval::new(start, end).map(Self::Closed),
      (None, Some(end)) => Err(IntervalError::EndWithoutStart { end }),
    }
  }

  /// Interval start τs, when known.
  pub fn start(&self) -> Option<Timestamp> {
    match self {
      Self::Closed(interval) => Some(interval.start()),
      Self::Open { start } => Some(*start),
      Self::Unknown => None,
    }
  }

  /// Interval end τe, when known.
  pub fn end(&self) -> Option<Timestamp> {
    match self {
      Self::Closed(interval) => Some(interval.end()),
      Self::Open { .. } | Self::Unknown => None,
    }
  }

  /// The closed span, when both bounds are known.
  pub fn closed(&self) -> Option<ClosedInterval> {
    match self {
      Self::Closed(interval) => Some(*interval),
      Self::Open { .. } | Self::Unknown => None,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::timestamp;

  #[test]
  fn converts_raw_parts_into_the_three_states() {
    assert_eq!(
      OccurrenceInterval::from_parts(None, None).expect("unknown"),
      OccurrenceInterval::Unknown
    );

    let start = timestamp(1_000);
    assert_eq!(
      OccurrenceInterval::from_parts(Some(start), None).expect("open"),
      OccurrenceInterval::Open { start }
    );

    let end = timestamp(2_000);
    assert_eq!(
      OccurrenceInterval::from_parts(Some(start), Some(end)).expect("closed"),
      OccurrenceInterval::Closed(ClosedInterval::new(start, end).expect("ordered"))
    );
  }

  #[test]
  fn rejects_reversed_and_end_only_bounds() {
    let start = timestamp(2_000);
    let end = timestamp(1_000);

    assert_eq!(
      OccurrenceInterval::from_parts(Some(start), Some(end)),
      Err(IntervalError::Reversed { start, end })
    );
    assert_eq!(
      OccurrenceInterval::from_parts(None, Some(end)),
      Err(IntervalError::EndWithoutStart { end })
    );
    assert_eq!(
      ClosedInterval::new(start, end),
      Err(IntervalError::Reversed { start, end })
    );
  }

  #[test]
  fn serde_round_trips_and_rejects_invalid_json() {
    let interval = OccurrenceInterval::Closed(
      ClosedInterval::new(timestamp(1_000), timestamp(2_000)).expect("ordered"),
    );
    let encoded = serde_json::to_string(&interval).expect("serialize");
    let decoded: OccurrenceInterval = serde_json::from_str(&encoded).expect("deserialize");
    assert_eq!(decoded, interval);

    let reversed = serde_json::json!({
      "closed": {
        "start": timestamp(2_000).to_rfc3339(),
        "end": timestamp(1_000).to_rfc3339(),
      }
    });
    assert!(
      serde_json::from_value::<OccurrenceInterval>(reversed).is_err(),
      "deserialization must not bypass the ordering invariant"
    );
  }
}
