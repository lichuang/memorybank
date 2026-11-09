//! Causal link construction from extraction output.
//!
//! Direction is **not normalized**: an edge keeps the relation reported at
//! extraction time. For example `A caused_by B` is stored as `A → B` with
//! `relation = caused_by` (B is the cause); traversal interprets the relation.
//! This preserves the model's wording and avoids inventing a canonical
//! direction for `enables` / `prevents`.

use std::collections::HashSet;

use crate::ports::NewLink;
use crate::types::{BankId, CausalRelation, ExtractedFact, FactId, LinkId, LinkType, Timestamp};

/// Maps extracted causal relations to directed `fact_link` rows.
///
/// `ids[i]` must be the persisted id of `extracted[i]` (the extraction pass
/// remaps indices after dropping invalid facts). Relations whose source or
/// target index cannot be resolved are dropped, as are self-references.
/// Duplicate `(from, to, relation)` triples are emitted once; base
/// `link_type` is `Causal`, `weight` is always `1.0` (the model-reported
/// `strength` deliberately does not enter the edge weight), and `now` stamps
/// `created_at`.
pub fn causal_links(
  bank: BankId,
  extracted: &[ExtractedFact],
  ids: &[FactId],
  now: Timestamp,
) -> Vec<NewLink> {
  let mut seen: HashSet<(FactId, FactId, CausalRelation)> = HashSet::new();
  let mut links = Vec::new();

  for (index, fact) in extracted.iter().enumerate() {
    let Some(&from) = ids.get(index) else {
      continue;
    };

    for causal in &fact.causal_relations {
      let Some(&to) = ids.get(causal.target_fact_index) else {
        continue;
      };
      if from == to {
        continue;
      }
      if !seen.insert((from, to, causal.relation)) {
        continue;
      }

      links.push(NewLink {
        id: LinkId::new(),
        bank_id: bank,
        from,
        to,
        link_type: LinkType::Causal,
        relation: Some(causal.relation),
        weight: 1.0,
        created_at: now,
      });
    }
  }

  links
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::timestamp;
  use crate::types::{CausalRef, FactType};

  fn extracted(what: &str, relations: Vec<CausalRef>) -> ExtractedFact {
    ExtractedFact {
      what: what.into(),
      when: String::new(),
      where_: String::new(),
      who: String::new(),
      why: String::new(),
      fact_type: FactType::World,
      confidence: None,
      occurred_start: None,
      occurred_end: None,
      mentioned_at: None,
      entities: Vec::new(),
      causal_relations: relations,
      context: None,
    }
  }

  fn relation(target_fact_index: usize, relation: CausalRelation) -> CausalRef {
    CausalRef {
      target_fact_index,
      relation,
      strength: 0.5,
    }
  }

  #[test]
  fn maps_one_way_causality() {
    let bank = BankId::new();
    let now = timestamp(1_000);
    let ids = [FactId::new(), FactId::new()];
    let facts = [
      extracted("a", vec![relation(1, CausalRelation::Causes)]),
      extracted("b", Vec::new()),
    ];

    let links = causal_links(bank, &facts, &ids, now);

    assert_eq!(links.len(), 1);
    assert_eq!(links[0].bank_id, bank);
    assert_eq!(links[0].from, ids[0]);
    assert_eq!(links[0].to, ids[1]);
    assert_eq!(links[0].link_type, LinkType::Causal);
    assert_eq!(links[0].relation, Some(CausalRelation::Causes));
    assert_eq!(links[0].weight, 1.0);
    assert_eq!(links[0].created_at, now);
  }

  #[test]
  fn keeps_the_reported_relation_direction() {
    let bank = BankId::new();
    let ids = [FactId::new(), FactId::new()];
    let facts = [
      extracted("a", vec![relation(1, CausalRelation::CausedBy)]),
      extracted("b", Vec::new()),
    ];

    let links = causal_links(bank, &facts, &ids, timestamp(1));

    assert_eq!(links.len(), 1);
    assert_eq!(links[0].from, ids[0], "the enclosing fact stays the source");
    assert_eq!(links[0].to, ids[1]);
    assert_eq!(links[0].relation, Some(CausalRelation::CausedBy));
  }

  #[test]
  fn maps_chained_causality() {
    let bank = BankId::new();
    let ids = [FactId::new(), FactId::new(), FactId::new()];
    let facts = [
      extracted("a", vec![relation(1, CausalRelation::Causes)]),
      extracted("b", vec![relation(2, CausalRelation::Enables)]),
      extracted("c", Vec::new()),
    ];

    let links = causal_links(bank, &facts, &ids, timestamp(1));

    assert_eq!(links.len(), 2);
    assert_eq!((links[0].from, links[0].to), (ids[0], ids[1]));
    assert_eq!((links[1].from, links[1].to), (ids[1], ids[2]));
    assert_eq!(links[1].relation, Some(CausalRelation::Enables));
  }

  #[test]
  fn drops_out_of_range_and_self_references() {
    let bank = BankId::new();
    let ids = [FactId::new(), FactId::new()];
    let facts = [
      extracted(
        "a",
        vec![
          relation(9, CausalRelation::Causes),
          relation(0, CausalRelation::Causes),
          relation(1, CausalRelation::Prevents),
        ],
      ),
      extracted("b", Vec::new()),
    ];

    let links = causal_links(bank, &facts, &ids, timestamp(1));

    assert_eq!(links.len(), 1);
    assert_eq!(links[0].to, ids[1]);
    assert_eq!(links[0].relation, Some(CausalRelation::Prevents));
  }

  #[test]
  fn deduplicates_identical_edges_but_keeps_distinct_relations() {
    let bank = BankId::new();
    let ids = [FactId::new(), FactId::new()];
    let facts = [
      extracted(
        "a",
        vec![
          relation(1, CausalRelation::Causes),
          relation(1, CausalRelation::Causes),
          relation(1, CausalRelation::Enables),
        ],
      ),
      extracted("b", Vec::new()),
    ];

    let links = causal_links(bank, &facts, &ids, timestamp(1));

    assert_eq!(
      links.len(),
      2,
      "identical triples collapse; relations differ"
    );
    assert_eq!(links[0].relation, Some(CausalRelation::Causes));
    assert_eq!(links[1].relation, Some(CausalRelation::Enables));
  }

  #[test]
  fn missing_ids_and_empty_input_produce_no_edges() {
    let bank = BankId::new();
    let ids = [FactId::new()];
    let facts = [extracted("a", vec![relation(1, CausalRelation::Causes)])];

    assert!(causal_links(bank, &facts, &ids, timestamp(1)).is_empty());
    assert!(causal_links(bank, &[], &[], timestamp(1)).is_empty());
  }
}
