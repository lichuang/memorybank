//! P1-5: atomic chunk claim, stale reclaim, and the monotonic claim token.

use std::time::Duration;

use engine::ports::{NewChunk, Store};
use engine::testing::timestamp;
use engine::types::{
  Bank, BankId, BankProfile, ChunkId, ChunkStatus, Source, SourceId, SourceStatus,
};
use store::SqliteStore;
use store::testing::memory_pool;

async fn setup() -> (SqliteStore, BankId, SourceId) {
  let store = SqliteStore::new(memory_pool().await.expect("in-memory store"));
  let bank = Bank {
    id: BankId::new(),
    name: "primary".into(),
    profile: BankProfile {
      background: "background".into(),
      skepticism: 3,
      literalism: 3,
      empathy: 3,
      bias_strength: 0.5,
    },
    created_at: timestamp(0),
    updated_at: timestamp(0),
  };
  store.create_bank(bank.clone()).await.expect("create bank");

  let source = Source {
    id: SourceId::new(),
    bank_id: bank.id,
    title: None,
    payload_text: "payload".into(),
    status: SourceStatus::Received,
    created_at: timestamp(0),
    updated_at: timestamp(0),
  };
  store
    .create_source(source.clone())
    .await
    .expect("create source");

  (store, bank.id, source.id)
}

fn chunk(bank: BankId, source: SourceId, index: u32, hash: &str, created_at: i64) -> NewChunk {
  NewChunk {
    id: ChunkId::new(),
    bank_id: bank,
    source_id: source,
    chunk_index: index,
    text: format!("text {index}"),
    content_hash: hash.into(),
    time_start: None,
    time_end: None,
    created_at: timestamp(created_at),
  }
}

#[tokio::test]
async fn claims_the_oldest_pending_chunk() {
  let (store, bank, source) = setup().await;
  store
    .append_chunks(
      source,
      vec![
        chunk(bank, source, 0, "hash-1", 1_000),
        chunk(bank, source, 1, "hash-2", 2_000),
      ],
    )
    .await
    .expect("seed chunks");

  let claimed = store
    .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
    .await
    .expect("claim")
    .expect("a chunk");

  assert_eq!(claimed.chunk.chunk_index, 0, "oldest first");
  assert_eq!(claimed.chunk.status, ChunkStatus::Extracting);
  assert_eq!(claimed.chunk.bank_id, bank);
  assert_eq!(claimed.chunk.source_id, source);
  assert_eq!(
    claimed.chunk.updated_at,
    timestamp(5_000),
    "claim token = now"
  );

  let next = store
    .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
    .await
    .expect("claim")
    .expect("second chunk");
  assert_eq!(next.chunk.chunk_index, 1);

  assert!(
    store
      .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
      .await
      .expect("claim")
      .is_none()
  );
}

#[tokio::test]
async fn returns_none_for_an_empty_bank() {
  let (store, bank, _source) = setup().await;

  assert!(
    store
      .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
      .await
      .expect("claim")
      .is_none()
  );
}

#[tokio::test]
async fn fresh_claims_are_not_stolen() {
  let (store, bank, source) = setup().await;
  store
    .append_chunks(source, vec![chunk(bank, source, 0, "hash-1", 1_000)])
    .await
    .expect("seed chunk");

  store
    .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
    .await
    .expect("first claim")
    .expect("chunk");

  assert!(
    store
      .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
      .await
      .expect("second claim")
      .is_none(),
    "a claim at the same instant is still fresh"
  );
}

#[tokio::test]
async fn stale_claims_are_reclaimed_with_a_larger_token() {
  let (store, bank, source) = setup().await;
  store
    .append_chunks(source, vec![chunk(bank, source, 0, "hash-1", 1_000)])
    .await
    .expect("seed chunk");

  let first = store
    .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
    .await
    .expect("first claim")
    .expect("chunk");
  assert_eq!(first.chunk.updated_at, timestamp(5_000));

  // now - stale_after = 5_500; the row (5_000) is stale and gets reclaimed.
  let second = store
    .claim_chunk(bank, timestamp(6_000), Duration::from_millis(500))
    .await
    .expect("reclaim")
    .expect("stale chunk");

  assert_eq!(second.chunk.id, first.chunk.id);
  assert_eq!(second.chunk.updated_at, timestamp(6_000));
  assert!(second.chunk.updated_at > first.chunk.updated_at);
}

#[tokio::test]
async fn same_millisecond_claims_get_distinct_tokens() {
  let (store, bank, source) = setup().await;
  store
    .append_chunks(source, vec![chunk(bank, source, 0, "hash-1", 1_000)])
    .await
    .expect("seed chunk");

  let first = store
    .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
    .await
    .expect("first claim")
    .expect("chunk");
  assert_eq!(first.chunk.updated_at, timestamp(5_000));

  // Simulate a release that keeps the timestamp, then claim again at the same
  // instant: `MAX(now, updated_at + 1)` must advance the token.
  sqlx::query("UPDATE chunk SET status = 'pending' WHERE id = ?")
    .bind(first.chunk.id.to_string())
    .execute(store.pool())
    .await
    .expect("manual release");

  let second = store
    .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
    .await
    .expect("second claim")
    .expect("chunk");

  assert_eq!(second.chunk.updated_at, timestamp(5_001));
  assert_ne!(first.chunk.updated_at, second.chunk.updated_at);
}

#[tokio::test]
async fn failed_chunks_are_never_claimed() {
  let (store, bank, source) = setup().await;
  let dead = chunk(bank, source, 0, "hash-1", 1_000);
  store
    .append_chunks(source, vec![dead.clone()])
    .await
    .expect("seed chunk");
  sqlx::query("UPDATE chunk SET status = 'failed' WHERE id = ?")
    .bind(dead.id.to_string())
    .execute(store.pool())
    .await
    .expect("dead-letter");

  assert!(
    store
      .claim_chunk(bank, timestamp(5_000), Duration::from_secs(3_600))
      .await
      .expect("claim")
      .is_none()
  );
}

#[tokio::test]
async fn concurrent_claims_grant_exactly_one_worker() {
  let (store, bank, source) = setup().await;
  store
    .append_chunks(source, vec![chunk(bank, source, 0, "hash-1", 1_000)])
    .await
    .expect("seed chunk");

  let mut handles = Vec::new();
  for _ in 0..10 {
    let store = store.clone();
    handles.push(tokio::spawn(async move {
      store
        .claim_chunk(bank, timestamp(5_000), Duration::ZERO)
        .await
        .expect("claim")
    }));
  }

  let mut granted = 0;
  for handle in handles {
    if handle.await.expect("join").is_some() {
      granted += 1;
    }
  }

  assert_eq!(granted, 1, "only one worker may own the chunk");
}
