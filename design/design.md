# memorybank Implementation Design (Rust)

Based on `../docs/hindsight.pdf` (the paper) and `memory-schema.md` (the finalized logical schema: 7 tables + §1.3 admission criteria).

## 0. Engineering Shape

Workspace layout (crate names carry no product prefix; `core` is avoided because it collides with Rust's built-in crate):

```
memorybank/                    # workspace root (virtual manifest; no root package)
├── crates/
│   ├── memorybank/            # facade: public API + wiring (depends on engine/store/llm)
│   │   ├── src/               # lib.rs (re-exports engine/llm/store)
│   │   ├── tests/             # workspace smoke tests
│   │   └── examples/          # quick-start examples (P5-8)
│   ├── engine/                # domain types + ports + orchestration of the three operations (core; no concrete backend dependency)
│   │   ├── types/             # Bank, Source, Chunk, Fact, Entity, FactEntity, FactLink...
│   │   ├── ports/             # Store, Llm, Embedder, Reranker, TemporalParser, Clock
│   │   ├── retain/            # inbox, chunking, extraction orchestration, tx2 commit, rescan
│   │   ├── recall/            # four channels + RRF + rerank + budget packing
│   │   ├── reflect/           # answer generation + opinion formation + reinforcement
│   │   ├── observe/           # observation synthesis + GC (work-list derivation)
│   │   ├── resolve/           # entity resolution ρ
│   │   └── link/              # temporal / semantic / causal link construction
│   ├── store/                 # sqlx: SQLite (default) / PostgreSQL (feature gate)
│   └── llm/                   # OpenAI-compatible chat / embedding provider
├── design/                    # design documents (three files)
└── docs/                      # the paper
```

| Crate | Responsibility | Depends on |
|---|---|---|
| `memorybank` (facade, `crates/memorybank`) | Facade: wires ports + public API + builders; the only direct dependency for users | engine + store + llm |
| `engine` | Domain types, port definitions, retain/recall/reflect orchestration | No concrete backend/model |
| `store` | Implements `Store` (SQLite/PG; indexes and query dialects converge here) | engine (ports) |
| `llm` | Implements `Llm` / `Embedder` (OpenAI-compatible HTTP) | engine (ports) |

Dependency direction is fixed at **facade → engine ← adapters**; engine never depends on an adapter in reverse. Users may implement any port themselves (custom storage, local models) to replace the default adapters.

## 1. Ports (Interface Contracts)

```rust
#[async_trait]
pub trait Llm: Send + Sync {
    /// Single entry point: structured output. Extraction (A.1/A.5.1), opinion
    /// formation (A.2), observation synthesis (A.3), Assess, and background
    /// merge all reuse it; prompts and JSON schemas are built in engine.
    async fn complete_json<T: DeserializeOwned + JsonSchema>(&self, req: &JsonRequest) -> Result<T>;
}

#[async_trait]
pub trait Embedder: Send + Sync {
    fn model(&self) -> &str;
    fn dim(&self) -> usize;
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;
}

pub trait TemporalParser: Send + Sync {
    fn parse(&self, query: &str, now: DateTime<Utc>) -> Option<TimeRange>;
}

pub trait Clock: Send + Sync {
    fn now(&self) -> DateTime<Utc>; // injectable for tests
}

#[async_trait]
pub trait Store: Send + Sync {
    // —— Inbox (schema §5.1, four stages) ——
    async fn create_source(&self, new: NewSource) -> Result<SourceId>;                        // tx0
    async fn append_chunks(&self, src: SourceId, chunks: Vec<NewChunk>) -> Result<usize>;    // tx1; hash conflicts skipped
    async fn claim_chunk(&self, bank: BankId) -> Result<Option<ClaimedChunk>>;               // atomic pending→extracting claim
    async fn commit_chunk(&self, commit: ChunkCommit) -> Result<()>;                         // tx2: all products + status=done atomically
    async fn release_chunk(&self, chunk: ChunkId) -> Result<()>;                             // on failure: back to pending

    // —— Read path (schema §5.4, four channels) ——
    async fn search_vector(&self, bank: BankId, q: &[f32], k: usize) -> Result<Vec<(FactId, f32)>>;
    async fn search_bm25(&self, bank: BankId, q: &str, k: usize) -> Result<Vec<(FactId, f32)>>;
    async fn neighbors(&self, bank: BankId, from: FactId) -> Result<Vec<Neighbor>>;          // fact_link + fact_entity-derived adjacency
    async fn facts_in_range(&self, bank: BankId, range: TimeRange) -> Result<Vec<FactId>>;
    async fn fetch_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>>;

    // —— Entity / opinion / observation / maintenance ——
    async fn upsert_entity(&self, new: NewEntity) -> Result<EntityId>;                       // handles UNIQUE(bank_id, canonical_name) conflicts
    async fn entity_candidates(&self, bank: BankId, mention: &str) -> Result<Vec<EntityCandidate>>;
    async fn insert_opinion(&self, op: NewOpinion) -> Result<FactId>;
    async fn update_opinion(&self, id: FactId, confidence: f32, text: Option<String>) -> Result<()>;
    async fn insert_observations(&self, obs: Vec<NewObservation>) -> Result<()>;
    async fn observations_of(&self, bank: BankId, entity: EntityId, limit: usize) -> Result<Vec<Fact>>;
    async fn pending_work(&self, bank: BankId) -> Result<WorkLists>;                        // derived work lists
    async fn reembed_batch(&self, bank: BankId, limit: usize) -> Result<usize>;
    async fn stats(&self, ...) -> ...;
}
```

Deliberate contract trade-offs (all map to the schema's §1.3 criteria):

1. **`commit_chunk` is the only transaction entry**: atomicity is guaranteed by the trait method; transaction objects never leak, so engine does not need to understand each backend's transaction API.
2. **`neighbors()` merges multiple sources inside the adapter**: `fact_link` rows + entity adjacency derived from `fact_entity` (the closure is never materialized) — criterion 3, "single source of truth".
3. **Work lists are always derived**: `pending_work` returns sources awaiting chunking, chunks pending/stale-claim, NULL embeddings, and entities whose evidence changed (`entity.last_seen > latest observation row's created_at`). There is no task-queue table — criterion 2, "strategies live at runtime".
4. **No model/threshold fields**: embedding model, chunking parameters, σt/θs/top-k, µ(ℓ) all live in configuration — criterion 2.
5. **`search_vector` must always be available**: ANN is adapter-internal optimization (SQLite brute-force cosine; pgvector indexing can be added later on PG); the contract only requires "return top-k".

## 2. Pipeline Implementation Notes

### 2.1 retain (Inbox + Extraction)

```
ingest(payload):
  canonical text → tx0: create_source
  chunking (session-alignment first / window fallback) → tx1: append_chunks (hash conflicts skipped)
  [sync mode] process this call's pending chunks immediately; [async mode] hand off to a worker
worker loop:
  claim_chunk → extract (Llm, A.1/A.5.1) → resolve (Store) → embed (Embedder)
             → build links (link module) → reinforce (Assess + confidence deltas)
             → commit_chunk (tx2)   // on failure: release_chunk, back to pending
rescan(): pending_work → re-chunk / re-extract (stale extracting judged by updated_at age)
```

- All model calls (extraction/Assess/embedding) happen **outside** tx2; tx2 is pure database writes.
- The first version defaults to **synchronous** (`retain` blocks until this call's chunks are done, schema §7.B#7); worker/rescan are separate APIs.

### 2.2 recall (Four Channels → RRF → Rerank → Budget)

- `search_vector` / `search_bm25` run concurrently with graph expansion and temporal filtering (`try_join!`).
- RRF: `Σ 1/(60 + rank_i)`, aggregated by fact_id.
- Rerank: `Reranker` scores the top-N candidates (identity stub in the first version).
- Budget packing: greedily accumulate tokens in final order (pluggable estimator) and truncate on overflow; hit rows `access_count++`.

### 2.3 reflect (Answer + Opinion Formation + Reinforcement)

- recall evidence → assemble the system message (name n, background h, verbalized φ(Θ) + β-dependent wording modulation).
- One `Llm::complete_json` call produces { answer text, candidate opinions[] } (A.2: first person, confidence, reasoning).
- New opinions go through `insert_opinion`; existing opinions are reinforced (Assess → c±α; strong contradictions may rewrite text).

### 2.4 observe (Observation Synthesis + GC)

- Work list: entities whose evidence changed (`entity.last_seen` later than their latest observation row) → `Llm` synthesizes 3–7 objective statements (A.3) → append-only `insert_observations` (full bindings).
- GC: by age / per-entity binding-count cap (optional in the first version).
- Scheduling: runtime sweeper (no persistent queue).

### 2.5 resolve (Entity Resolution)

- Normalize the mention → exact hit on `entity.canonical_name` (index lookup); on miss, use the three-feature ρ (string / co-occurrence / temporal); variants are derived from `fact_entity.mention_text` (cached in memory per call, no cache table).
- Persist: `upsert_entity` (reuse on UNIQUE conflict) + binding rows + entity statistics deltas (mention_count / first_seen / last_seen, same transaction).

### 2.6 link (Link Construction)

- **temporal**: interval distance → `exp(−Δt/σt)`; prune per pair within the window; write two rows as a pair.
- **semantic**: after embeddings are ready, ANN top-k (budget predicate), **directed single row** (from = selecting side).
- **causal**: extraction output `causal_relations[]` (same chunk only), normalized to "cause→effect", single row.
- Link thresholds and caps all come from configuration.

## 3. Cross-Cutting Concerns

| Concern | Approach |
|---|---|
| Async runtime | tokio; ports use `async_trait` (dyn-compatible, injectable at runtime) |
| Errors | `thiserror`; `engine::Error` with five variants: `Retryable` (LLM timeouts, DB contention) / `Fatal` (invalid model output, data corruption) / `Config` (invalid configuration or caller settings) / `NotFound` / `Conflict`; workers retry only `Retryable` |
| Configuration | serde structs: bank profile (Θ/β), model choices, thresholds, chunking parameters, concurrency |
| Observability | `tracing` spans: per-chunk latency, LLM calls/retries, per-channel recall hits, budget usage |
| IDs/Time | UUIDv7; chrono UTC |
| Vectors | in-memory `Vec<f32>`; validate dimensions on write; adapters own the persistence form |
| Concurrency | configurable chunk-worker count; claims guarantee single execution per chunk; no global lock per bank |

## 4. Testing Strategy

1. **Deterministic fakes**: `FakeLlm` (returns fixed structured output per input), `HashEmbedder` (deterministic character-hash vectors), `IdentityReranker` — the pipeline is testable end to end.
2. **Real SQLite**: an isolated in-memory database per test + migrations; covers idempotency (re-ingest/incremental/crash recovery), transaction rollback, and the two-row invariant.
3. **Contract tests**: the same behavioral suite for `Store` runs against both SQLite and PG (test kit in the `store` crate).
4. **Evaluation (deferred)**: LongMemEval / LoCoMo harness with real models (P5).

## 5. Phasing

| Phase | Content | Exit criteria |
|---|---|---|
| **P0** | types + ports + SQLite migration (physical schema) + fakes | compiles; contract-test skeleton runnable |
| **P1** | retain loop: inbox → chunking → extraction → tx2 → rescan | re-ingest / interruption-recovery tests all green |
| **P2** | recall loop: four channels + RRF + rerank stub + budget | hit rate on a fixed corpus meets expectations |
| **P3** | reflect: answer + opinion formation + reinforcement | profile modulation observable; opinions can evolve |
| **P4** | observe + resolve/link polish | observations synthesized/reclaimed; resolution quality measurable |
| **P5** | PG/pgvector adapter, real reranker, TemporalParser, evaluation | dual-backend contract tests pass; benchmarks runnable |

## 6. Open Items

1. `Store` shape: single trait (current draft) vs capability sub-traits (read/write/maintenance) — settle in P0.
2. Whether retain's default API is synchronous or asynchronous (schema §7.B#7 recommends synchronous first).
3. First implementations of reranker / temporal parser (identity stub + rule-based parser first; model-based versions deferred).
4. Tokenization for chunking and the budget estimator (do we reuse the embedding model's tokenizer?).
5. Evaluation integration timing and harness shape (P5).
6. Semantic follow-ups inherited from the schema doc: `§7.A#2` (nullable confidence on extraction-produced opinion rows vs an initial confidence).
