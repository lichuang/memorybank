//! Lexical (BM25) recall channel (paper §4.2.2, eq. 11).
//!
//! SQLite has no word-level BM25/GIN equivalent; `fact_fts` uses the FTS5
//! trigram tokenizer, so this channel is substring-oriented. Terms shorter than
//! three characters cannot match a trigram and are ignored. The store owns the
//! index and ranking; this module is the recall-level policy.

use crate::Result;
use crate::ports::{FactScope, Store};
use crate::types::{BankId, FactId};

/// BM25-channel parameters. Runtime configuration — never persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct Bm25Config {
  /// Number of candidates requested from the store. `0` disables the channel.
  pub top_k: usize,
}

impl Default for Bm25Config {
  fn default() -> Self {
    Self { top_k: 50 }
  }
}

/// Returns the top lexical matches in `scope`, highest BM25 score first.
///
/// An empty/whitespace-only query returns an empty list without a store call.
/// The store handles tokenization and the trigram limitation; results are
/// already ranked and truncated to `top_k`.
pub async fn bm25_channel(
  store: &dyn Store,
  bank: BankId,
  query: &str,
  cfg: &Bm25Config,
  scope: &FactScope,
) -> Result<Vec<(FactId, f32)>> {
  if cfg.top_k == 0 || query.trim().is_empty() {
    return Ok(Vec::new());
  }

  let mut results = store.search_bm25(bank, query, cfg.top_k, scope).await?;
  results.truncate(cfg.top_k);
  Ok(results)
}
