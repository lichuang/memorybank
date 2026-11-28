//! The reflect entry point: recall → prompt → LLM → persist opinions.
//!
//! This is the facade-level orchestration. It deliberately reuses the same
//! building blocks as the lower layers: `system_message` for the prompt,
//! `reflect_llm` for the model call, `prepare_opinion` + `insert_opinion` for
//! persistence. Nothing here forks the reinforcement path (P3-5 is still on hold),
//! so an opinion is written once and later calls only read it through recall.

use engine::ports::{Embedder, now_ts};
use engine::recall::RecallOptions;
use engine::reflect::{
  ReflectConfig, ReflectResult, SystemMessageConfig, prepare_opinion, reflect_llm, system_message,
};
use engine::retain::embed_facts;

use crate::retain::MemoryBank;
use engine::types::{BankId, FactId};
use engine::{Error, Result};

/// Options for one reflect call.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflectOptions {
  /// Recall settings used to gather evidence.
  pub recall: RecallOptions,
  /// System-message caps.
  pub message: SystemMessageConfig,
  /// Model call parameters.
  pub llm: ReflectConfig,
  /// Whether to persist the opinions the model forms.
  pub persist_opinions: bool,
}

impl Default for ReflectOptions {
  fn default() -> Self {
    Self {
      recall: RecallOptions::default(),
      message: SystemMessageConfig::default(),
      llm: ReflectConfig::default(),
      persist_opinions: true,
    }
  }
}

/// Runs a reflect call: retrieve evidence, ask the model, store new opinions.
///
/// The flow mirrors the paper's four steps (§5.4.2): recall the query, assemble
/// the system message from the bank profile and the evidence, run one structured
/// LLM call, then persist the opinions it returned.
///
/// Opinion ids are allocated **before** the insert, so a retry after a transport
/// error cannot duplicate rows: the failed attempt wrote nothing, and the retry
/// generates fresh ids for a fresh attempt. Each opinion is inserted in its own
/// transaction; if one insert fails, the opinions already written stay (they are
/// valid rows) and the error is returned, so the caller can decide whether to
/// retry.
///
/// Access counters for the recalled evidence follow the same best-effort rule as
/// `recall`: a counter failure never fails reflect.
impl MemoryBank {
  /// Reflects on `query` for `bank`.
  pub async fn reflect(
    &self,
    bank: BankId,
    query: &str,
    options: ReflectOptions,
  ) -> Result<ReflectResult> {
    if query.trim().is_empty() {
      return Err(Error::config("reflect query must not be empty"));
    }

    let llm = self
      .llm
      .as_ref()
      .ok_or_else(|| Error::config("reflect requires an Llm"))?;
    let embedder = self
      .embedder
      .as_ref()
      .ok_or_else(|| Error::config("reflect requires an Embedder"))?;

    let recalled = self.recall(bank, query, options.recall.clone()).await?;
    let bank_row = self.store.load_bank(bank).await?;
    let system = system_message(&bank_row, &recalled.facts, &options.message);

    let response = reflect_llm(llm.as_ref(), system, query, &options.llm).await?;

    let opinions_formed = if options.persist_opinions {
      self
        .persist_opinions(bank, &response.opinions, embedder.as_ref())
        .await?
    } else {
      Vec::new()
    };

    Ok(ReflectResult {
      answer: response.answer,
      opinions_formed,
      recalled,
    })
  }

  /// Writes one row per draft, returning the ids that were written.
  async fn persist_opinions(
    &self,
    bank: BankId,
    drafts: &[engine::types::OpinionDraft],
    embedder: &dyn Embedder,
  ) -> Result<Vec<FactId>> {
    if drafts.is_empty() {
      return Ok(Vec::new());
    }

    let now = now_ts(self.clock.as_ref());
    let texts: Vec<String> = drafts.iter().map(|draft| draft.opinion.clone()).collect();
    // Embedding degradation is not fatal here either: an opinion without a
    // vector is still a valid row and is backfilled later.
    let embeddings = embed_facts(embedder, &texts, self.pipeline.embedding_batch).await?;

    let mut ids = Vec::with_capacity(drafts.len());
    for (draft, embedding) in drafts.iter().zip(embeddings) {
      let id = FactId::new();
      let embedding = embedding.into_option();
      let opinion = prepare_opinion(
        self.store.as_ref(),
        id,
        bank,
        &draft.opinion,
        draft.confidence,
        embedding,
        Some(&draft.reasoning),
        now,
      )
      .await?;
      self.store.insert_opinion(opinion).await?;
      ids.push(id);
    }
    Ok(ids)
  }

  /// Like [`MemoryBank::reflect`] but with default options.
  pub async fn reflect_default(&self, bank: BankId, query: &str) -> Result<ReflectResult> {
    self.reflect(bank, query, ReflectOptions::default()).await
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_options_persist_opinions() {
    let options = ReflectOptions::default();
    assert!(options.persist_opinions);
    assert_eq!(
      options.recall.token_budget,
      RecallOptions::default().token_budget
    );
  }
}
