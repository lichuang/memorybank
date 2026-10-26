//! Deterministic test doubles for the ports.
//!
//! These are shipped with the engine (not behind a `cfg(test)`) because adapter
//! and facade integration tests need them too. None of them touch the network or
//! the system clock, so unit tests stay hermetic.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{Duration, NaiveDate};
use serde_json::Value;

use crate::ports::{Clock, Embedder, JsonRequest, Llm, Reranker, TemporalParser};
use crate::types::{TimeRange, Timestamp};
use crate::{Error, Result};

/// A scripted [`Llm`]: returns queued JSON responses per schema and records
/// every request.
///
/// Requests are keyed by [`JsonRequest::schema_name`], so one fake can drive
/// extraction, reflection, observation synthesis, and assessment in the same
/// test. The queue is consumed in FIFO order; an exhausted (or unknown) schema
/// yields [`Error::Fatal`], mirroring an adapter that cannot recover.
#[derive(Debug, Default)]
pub struct FakeLlm {
  scripted: Mutex<HashMap<&'static str, VecDeque<Value>>>,
  calls: Mutex<Vec<JsonRequest>>,
}

impl FakeLlm {
  /// Creates an empty fake.
  pub fn new() -> Self {
    Self::default()
  }

  /// Queues one response for `schema_name`.
  pub fn push(&self, schema_name: &'static str, value: Value) {
    self
      .scripted
      .lock()
      .expect("fake llm script mutex poisoned")
      .entry(schema_name)
      .or_default()
      .push_back(value);
  }

  /// Returns every request received so far, in call order.
  ///
  /// Useful for asserting prompt content (e.g. profile verbalization) without
  /// reaching into adapter internals.
  pub fn calls(&self) -> Vec<JsonRequest> {
    self
      .calls
      .lock()
      .expect("fake llm calls mutex poisoned")
      .clone()
  }
}

#[async_trait]
impl Llm for FakeLlm {
  async fn complete_json_raw(&self, req: JsonRequest) -> Result<Value> {
    let schema_name = req.schema_name;
    self
      .calls
      .lock()
      .expect("fake llm calls mutex poisoned")
      .push(req);

    self
      .scripted
      .lock()
      .expect("fake llm script mutex poisoned")
      .get_mut(schema_name)
      .and_then(VecDeque::pop_front)
      .ok_or_else(|| {
        Error::fatal(format!(
          "FakeLlm has no scripted response for schema `{schema_name}`"
        ))
      })
  }

  fn model(&self) -> &str {
    "fake-llm"
  }
}

/// A deterministic character-trigram hash embedder.
///
/// Identical text always produces an identical, L2-normalized vector of length
/// [`HashEmbedder::dim`]; an empty text yields a zero vector. This is a test
/// double, not a semantic model: collisions are expected.
#[derive(Debug, Clone, Copy)]
pub struct HashEmbedder {
  dim: usize,
}

impl HashEmbedder {
  /// Creates an embedder with `dim` dimensions.
  ///
  /// # Panics
  ///
  /// Panics when `dim` is zero.
  pub fn new(dim: usize) -> Self {
    assert!(dim > 0, "embedding dimension must be positive");
    Self { dim }
  }
}

#[async_trait]
impl Embedder for HashEmbedder {
  fn model(&self) -> &str {
    "hash-embedder"
  }

  fn dim(&self) -> usize {
    self.dim
  }

  async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    Ok(texts.iter().map(|text| self.embed_one(text)).collect())
  }
}

impl HashEmbedder {
  fn embed_one(&self, text: &str) -> Vec<f32> {
    let mut vector = vec![0.0_f32; self.dim];
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
      return vector;
    }

    let grams: Vec<&[char]> = if chars.len() < 3 {
      vec![chars.as_slice()]
    } else {
      chars.windows(3).collect()
    };

    for gram in grams {
      let gram: String = gram.iter().collect();
      let hash = fnv1a(gram.as_bytes());
      let index = (hash % self.dim as u64) as usize;
      let sign = if hash >> 63 == 0 { 1.0 } else { -1.0 };
      vector[index] += sign;
    }

    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
      for value in &mut vector {
        *value /= norm;
      }
    }

    vector
  }
}

fn fnv1a(bytes: &[u8]) -> u64 {
  let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
  for byte in bytes {
    hash ^= u64::from(*byte);
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
  }
  hash
}

/// A reranker that preserves the input order.
///
/// Scores are strictly decreasing (`1.0`, `(n-1)/n`, …) so the order survives any
/// sort, not only a stable one.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityReranker;

#[async_trait]
impl Reranker for IdentityReranker {
  async fn rerank(&self, _query: &str, docs: &[&str]) -> Result<Vec<f32>> {
    let len = docs.len();
    Ok(
      (0..len)
        .map(|index| (len - index) as f32 / len as f32)
        .collect(),
    )
  }
}

/// A small rule-based [`TemporalParser`].
///
/// Understands ISO dates (`2023-05-01`), `today`, `yesterday`, `last week`
/// (previous 7 days), `last month` (previous 30 days), and `last year` (previous
/// 365 days). Relative windows end at `now`; absolute dates cover the whole UTC
/// day. Anything else returns `None` — no guessing.
#[derive(Debug, Clone, Copy, Default)]
pub struct RuleTemporalParser;

impl TemporalParser for RuleTemporalParser {
  fn parse(&self, query: &str, now: Timestamp) -> Option<TimeRange> {
    let lower = query.to_lowercase();

    if let Some(date) = find_iso_date(&lower) {
      return Some(day_range(date));
    }

    if lower.contains("today") {
      return Some(day_range(now.date_naive()));
    }
    if lower.contains("yesterday") {
      return Some(day_range(now.date_naive() - Duration::days(1)));
    }
    if lower.contains("last week") {
      return Some(TimeRange {
        start: now - Duration::days(7),
        end: now,
      });
    }
    if lower.contains("last month") {
      return Some(TimeRange {
        start: now - Duration::days(30),
        end: now,
      });
    }
    if lower.contains("last year") {
      return Some(TimeRange {
        start: now - Duration::days(365),
        end: now,
      });
    }

    None
  }
}

fn find_iso_date(query: &str) -> Option<NaiveDate> {
  query.split_whitespace().find_map(|token| {
    let token = token.trim_matches(|c: char| !c.is_ascii_digit() && c != '-');
    NaiveDate::parse_from_str(token, "%Y-%m-%d").ok()
  })
}

fn day_range(date: NaiveDate) -> TimeRange {
  let start = date
    .and_hms_opt(0, 0, 0)
    .expect("midnight is a valid time")
    .and_utc();
  TimeRange {
    start,
    end: start + Duration::days(1) - Duration::milliseconds(1),
  }
}

/// A clock frozen at a fixed instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedClock(pub Timestamp);

impl Clock for FixedClock {
  fn now(&self) -> Timestamp {
    self.0
  }
}

#[cfg(test)]
mod tests {
  use chrono::TimeZone;
  use serde_json::json;

  use super::*;
  use crate::ports::LlmExt;
  use crate::types::FactExtractionResponse;

  fn request(schema_name: &'static str, user: &str) -> JsonRequest {
    JsonRequest {
      system: None,
      user: user.into(),
      schema_name,
      schema: json!({ "type": "object" }),
      temperature: 0.0,
      max_tokens: 256,
    }
  }

  #[tokio::test]
  async fn fake_llm_pops_scripted_responses_in_order() {
    let llm = FakeLlm::new();
    llm.push("fact_extraction", json!({ "facts": [] }));
    llm.push("fact_extraction", json!({ "facts": [{ "what": "x" }] }));

    let first = llm
      .complete_json_raw(request("fact_extraction", "first"))
      .await
      .expect("first response");
    let second = llm
      .complete_json_raw(request("fact_extraction", "second"))
      .await
      .expect("second response");

    assert_eq!(first, json!({ "facts": [] }));
    assert_eq!(second["facts"][0]["what"], json!("x"));

    let calls = llm.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].user, "first");
    assert_eq!(calls[1].user, "second");
    assert_eq!(calls[0].schema_name, "fact_extraction");
    assert_eq!(llm.model(), "fake-llm");
  }

  #[tokio::test]
  async fn fake_llm_separates_schemas() {
    let llm = FakeLlm::new();
    llm.push("fact_extraction", json!({ "facts": [] }));
    llm.push("reflect", json!({ "answer": "ok", "opinions": [] }));

    let extraction = llm
      .complete_json_raw(request("fact_extraction", "extract"))
      .await
      .expect("extraction");
    let reflection = llm
      .complete_json_raw(request("reflect", "reflect"))
      .await
      .expect("reflection");

    assert_eq!(extraction, json!({ "facts": [] }));
    assert_eq!(reflection["answer"], json!("ok"));
  }

  #[tokio::test]
  async fn fake_llm_exhaustion_is_fatal() {
    let llm = FakeLlm::new();

    let error = llm
      .complete_json_raw(request("fact_extraction", "nothing queued"))
      .await
      .expect_err("exhausted queue");

    assert!(matches!(error, Error::Fatal(_)));
    assert!(!error.is_retryable());
    assert!(error.to_string().contains("fact_extraction"));
    assert_eq!(llm.calls().len(), 1, "the failed call is still recorded");
  }

  #[tokio::test]
  async fn fake_llm_works_through_the_typed_extension() {
    let fake = FakeLlm::new();
    fake.push("fact_extraction", json!({ "facts": [] }));

    let llm: Box<dyn Llm> = Box::new(fake);
    let response: FactExtractionResponse = llm
      .complete_json(request("fact_extraction", "extract"))
      .await
      .expect("typed response");

    assert!(response.facts.is_empty());
  }

  #[tokio::test]
  async fn hash_embedder_is_deterministic_and_normalized() {
    let embedder = HashEmbedder::new(16);

    let first = embedder
      .embed(&["Emily works at Google."])
      .await
      .expect("embed");
    let second = embedder
      .embed(&["Emily works at Google."])
      .await
      .expect("embed");
    let other = embedder
      .embed(&["The weather is cold."])
      .await
      .expect("embed");

    assert_eq!(first, second, "same input, same vector");
    assert_ne!(first, other, "different text should differ");
    assert_eq!(first[0].len(), 16);
    assert_eq!(embedder.dim(), 16);
    assert_eq!(embedder.model(), "hash-embedder");

    let norm = first[0].iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
  }

  #[tokio::test]
  async fn hash_embedder_handles_empty_input() {
    let embedder = HashEmbedder::new(8);

    let vectors = embedder.embed(&[]).await.expect("empty batch");
    assert!(vectors.is_empty());

    let vectors = embedder.embed(&[""]).await.expect("empty text");
    assert_eq!(vectors, vec![vec![0.0; 8]]);
  }

  #[tokio::test]
  async fn identity_reranker_preserves_order() {
    let reranker: Box<dyn Reranker> = Box::new(IdentityReranker);

    let scores = reranker
      .rerank("query", &["a", "b", "c"])
      .await
      .expect("rerank");

    assert_eq!(scores.len(), 3);
    assert!(scores.windows(2).all(|pair| pair[0] > pair[1]));
  }

  #[test]
  fn rule_parser_reads_absolute_dates() {
    let now = chrono::Utc.with_ymd_and_hms(2024, 6, 9, 12, 0, 0).unwrap();
    let range = RuleTemporalParser
      .parse("what happened on 2023-05-01?", now)
      .expect("ISO date");

    assert_eq!(range.start.to_rfc3339(), "2023-05-01T00:00:00+00:00");
    assert_eq!(range.end.to_rfc3339(), "2023-05-01T23:59:59.999+00:00");
  }

  #[test]
  fn rule_parser_reads_relative_windows() {
    let now = chrono::Utc.with_ymd_and_hms(2024, 6, 9, 12, 0, 0).unwrap();
    let parser = RuleTemporalParser;

    let today = parser.parse("what happened today?", now).expect("today");
    assert_eq!(today.start.to_rfc3339(), "2024-06-09T00:00:00+00:00");
    assert_eq!(today.end.to_rfc3339(), "2024-06-09T23:59:59.999+00:00");

    let yesterday = parser
      .parse("who did I meet yesterday", now)
      .expect("yesterday");
    assert_eq!(yesterday.start.to_rfc3339(), "2024-06-08T00:00:00+00:00");

    let last_week = parser.parse("LAST WEEK", now).expect("last week");
    assert_eq!(last_week.start, now - Duration::days(7));
    assert_eq!(last_week.end, now);

    assert!(parser.parse("who is Emily?", now).is_none());
  }

  #[test]
  fn fixed_clock_returns_its_instant() {
    let now = chrono::Utc.with_ymd_and_hms(2024, 6, 9, 12, 0, 0).unwrap();
    let clock: Box<dyn Clock> = Box::new(FixedClock(now));

    assert_eq!(clock.now(), now);
  }
}
