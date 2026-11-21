//! Latency baseline for exact vector search.
//!
//! Run with `cargo bench -p store --bench vector_search`. Each case builds an
//! isolated in-memory bank and measures `Store::search_vector` at 1k and 10k
//! embedded facts, both across every network and restricted to two networks —
//! the latter scans once per listed type. Numbers are reference data, not a
//! threshold, and CI does not run benches.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use engine::ports::{FactScope, Store};
use engine::types::{Bank, BankId, BankProfile, FactId, FactType};
use store::testing::memory_pool;
use store::{EMBEDDING_DIM, SqliteStore};

fn vector(seed: usize) -> Vec<f32> {
  let mut values = vec![0.0_f32; EMBEDDING_DIM];
  // A few non-zero components keep cosine meaningful while the vectors stay
  // cheap to build.
  for offset in 0..8 {
    values[(seed * 7 + offset * 31) % EMBEDDING_DIM] = ((seed + offset) % 17) as f32 + 1.0;
  }
  values
}

fn blob(values: &[f32]) -> Vec<u8> {
  values
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect()
}

async fn bank_with_facts(size: usize) -> (SqliteStore, BankId) {
  let pool = memory_pool().await.expect("pool");
  let store = SqliteStore::new(pool.clone());
  let bank = Bank {
    id: BankId::new(),
    name: "bench".into(),
    profile: BankProfile {
      background: "bench".into(),
      skepticism: 3,
      literalism: 3,
      empathy: 3,
      bias_strength: 0.5,
    },
    created_at: engine::testing::timestamp(0),
    updated_at: engine::testing::timestamp(0),
  };
  store.create_bank(bank.clone()).await.expect("bank");

  let mut tx = pool.begin().await.expect("tx");
  for index in 0..size {
    let id = FactId::new();
    let fact_type = if index % 4 == 0 {
      FactType::Opinion
    } else {
      FactType::World
    };
    let confidence = (fact_type == FactType::Opinion).then_some(0.5_f64);
    sqlx::query(
      "INSERT INTO fact (id, bank_id, fact_type, text, confidence, created_at, updated_at) \
       VALUES (?, ?, ?, 'bench fact', ?, 0, 0)",
    )
    .bind(id.to_string())
    .bind(bank.id.to_string())
    .bind(fact_type.as_str())
    .bind(confidence)
    .execute(&mut *tx)
    .await
    .expect("fact");

    sqlx::query(
      "INSERT INTO fact_vec(rowid, fact_id, bank_id, fact_type, embedding) \
       SELECT rowid, id, bank_id, fact_type, ? FROM fact WHERE id = ?",
    )
    .bind(blob(&vector(index)))
    .bind(id.to_string())
    .execute(&mut *tx)
    .await
    .expect("vector");
  }
  tx.commit().await.expect("commit");

  (store, bank.id)
}

fn vector_search(c: &mut Criterion) {
  let runtime = tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()
    .expect("runtime");

  for size in [1_000_usize, 10_000] {
    let (store, bank) = runtime.block_on(bank_with_facts(size));
    let query = vector(size / 2);

    let mut group = c.benchmark_group(format!("vector_search/{size}"));
    group.bench_function(BenchmarkId::new("all", size), |b| {
      b.to_async(&runtime).iter(|| async {
        let hits = store
          .search_vector(bank, black_box(&query), 10, &FactScope::All)
          .await
          .expect("search");
        black_box(hits.len());
      });
    });
    group.bench_function(BenchmarkId::new("two_types", size), |b| {
      b.to_async(&runtime).iter(|| async {
        // The scoped path issues one KNN per listed type.
        let hits = store
          .search_vector(
            bank,
            black_box(&query),
            10,
            &FactScope::Types(vec![FactType::World, FactType::Opinion]),
          )
          .await
          .expect("search");
        black_box(hits.len());
      });
    });
    group.finish();
  }
}

criterion_group!(benches, vector_search);
criterion_main!(benches);
