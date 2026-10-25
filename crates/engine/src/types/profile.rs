//! The agent profile P = (n, Θ, h).

use std::fmt;

use serde::{Deserialize, Serialize};

/// Behavioral profile of a memory bank (paper §5.2–§5.3).
///
/// Disposition dimensions range over 1..=5 and bias strength over [0, 1]. Call
/// [`BankProfile::validate`] before persisting a profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BankProfile {
  /// Short background description h, written in the first person.
  pub background: String,
  /// Skepticism S: 1 = trusting, 5 = skeptical.
  pub skepticism: u8,
  /// Literalism L: 1 = flexible, 5 = literal.
  pub literalism: u8,
  /// Empathy E: 1 = detached, 5 = empathetic.
  pub empathy: u8,
  /// Bias strength β in [0, 1]; controls how strongly preferences shape opinions.
  pub bias_strength: f32,
}

/// Error returned when a profile contains an out-of-range value.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileError {
  /// Name of the invalid field.
  pub field: &'static str,
  /// The rejected value.
  pub value: f32,
}

impl fmt::Display for ProfileError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "invalid {}: {} is out of range", self.field, self.value)
  }
}

impl std::error::Error for ProfileError {}

impl BankProfile {
  /// Validates all profile dimensions.
  ///
  /// # Errors
  ///
  /// Returns [`ProfileError`] for any disposition outside 1..=5 or a bias
  /// strength outside [0, 1] (NaN is rejected).
  pub fn validate(&self) -> Result<(), ProfileError> {
    check_level("skepticism", self.skepticism)?;
    check_level("literalism", self.literalism)?;
    check_level("empathy", self.empathy)?;

    if (0.0..=1.0).contains(&self.bias_strength) {
      Ok(())
    } else {
      Err(ProfileError {
        field: "bias_strength",
        value: self.bias_strength,
      })
    }
  }
}

fn check_level(field: &'static str, value: u8) -> Result<(), ProfileError> {
  if (1..=5).contains(&value) {
    Ok(())
  } else {
    Err(ProfileError {
      field,
      value: f32::from(value),
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn sample_profile() -> BankProfile {
    BankProfile {
      background: "I am a careful assistant.".into(),
      skepticism: 3,
      literalism: 3,
      empathy: 3,
      bias_strength: 0.2,
    }
  }

  #[test]
  fn accepts_boundary_values() {
    for level in [1, 5] {
      let profile = BankProfile {
        skepticism: level,
        literalism: level,
        empathy: level,
        ..sample_profile()
      };
      assert!(profile.validate().is_ok());
    }

    for bias in [0.0, 1.0] {
      let profile = BankProfile {
        bias_strength: bias,
        ..sample_profile()
      };
      assert!(profile.validate().is_ok());
    }
  }

  #[test]
  fn rejects_out_of_range_levels() {
    for value in [0, 6] {
      let profile = BankProfile {
        skepticism: value,
        ..sample_profile()
      };
      assert_eq!(
        profile.validate().expect_err("bad skepticism").field,
        "skepticism"
      );

      let profile = BankProfile {
        literalism: value,
        ..sample_profile()
      };
      assert_eq!(
        profile.validate().expect_err("bad literalism").field,
        "literalism"
      );

      let profile = BankProfile {
        empathy: value,
        ..sample_profile()
      };
      assert_eq!(
        profile.validate().expect_err("bad empathy").field,
        "empathy"
      );
    }
  }

  #[test]
  fn rejects_out_of_range_bias_strength() {
    for bias in [-0.1, 1.1, f32::NAN] {
      let profile = BankProfile {
        bias_strength: bias,
        ..sample_profile()
      };
      let error = profile.validate().expect_err("invalid bias strength");
      assert_eq!(error.field, "bias_strength");
    }
  }
}
