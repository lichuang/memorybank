//! Injectable clock port.

use crate::types::Timestamp;

/// Wall clock; injected so tests and replay can control time.
///
/// Library code must never call `Utc::now()` directly.
pub trait Clock: Send + Sync {
  /// Current UTC time.
  fn now(&self) -> Timestamp;
}

/// Current time truncated to the schema's millisecond precision.
///
/// The logical `TS` type is millisecond-precision (schema §2), while chrono
/// timestamps carry nanoseconds. Routing every write through this helper keeps
/// stored values canonical.
pub fn now_ts(clock: &dyn Clock) -> Timestamp {
  Timestamp::from_timestamp_millis(clock.now().timestamp_millis())
    .expect("a timestamp derived from a DateTime is always representable")
}

#[cfg(test)]
mod tests {
  use super::*;

  struct TestClock(Timestamp);

  impl Clock for TestClock {
    fn now(&self) -> Timestamp {
      self.0
    }
  }

  #[test]
  fn now_ts_truncates_to_milliseconds() {
    let raw = Timestamp::from_timestamp(1_700_000_000, 123_456_789).expect("valid timestamp");
    let clock = TestClock(raw);

    let ts = now_ts(&clock);

    assert_eq!(ts.timestamp_subsec_nanos(), 123_000_000);
    assert_eq!(ts.timestamp_millis(), raw.timestamp_millis());
  }

  #[test]
  fn dyn_clock_is_injectable() {
    let now = Timestamp::from_timestamp_millis(1_700_000_000_000).expect("valid timestamp");
    let clock: Box<dyn Clock> = Box::new(TestClock(now));

    assert_eq!(now_ts(clock.as_ref()), now);
  }
}
