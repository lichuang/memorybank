//! Time primitives shared by the domain model and the ports.

use serde::{Deserialize, Serialize};

/// A UTC timestamp. All wall-clock values in the system are UTC.
pub type Timestamp = chrono::DateTime<chrono::Utc>;

/// A closed time interval in UTC, used as the temporal-retrieval window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
  /// Inclusive start.
  pub start: Timestamp,
  /// Inclusive end.
  pub end: Timestamp,
}
