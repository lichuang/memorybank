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
│   ├── store/                 # sqlx: SQLite adapter
│   └── llm/                   # HTTP-compatible provider + optional local adapters
├── design/                    # design documents (three files)
└── docs/                      # the paper
```

| Crate | Responsibility | Depends on |
|---|---|---|
| `memorybank` (facade, `crates/memorybank`) | Facade: wires ports + public API + builders; the only direct dependency for users | engine + store + llm |
| `engine` | Domain types, port definitions, retain/recall/reflect orchestration | No concrete backend/model |
| `store` | Implements `Store` on SQLite (query dialects stay here) | engine (ports) |
| `llm` | Implements `Llm` / `Embedder` (OpenAI-compatible HTTP) | engine (ports) |

Dependency direction is fixed at **facade → engine ← adapters**; engine never depends on an adapter in reverse. Users may implement any port themselves (custom storage, local models) to replace the default adapters.

> **Backend scope (current): SQLite only.** PostgreSQL (and pgvector) is deliberately deferred: there is no `postgres` feature, module, or migration today. The logical schema and the `Store` contract stay backend-neutral so a second adapter can be added later without touching the engine; revisit when multi-backend deployment or a read-only serving replica is actually needed.

## 1. Ports (Interface Contracts)

```rust
#[async_trait]
pub trait Llm: Send + Sync {
    /// Dyn-safe transport: returns raw JSON. Extraction (A.1/A.5.1), opinion
    /// formation (A.2), observation synthesis (A.3), Assess, and background
    /// merge all reuse it; prompts and JSON schemas are built in engine.
    async fn complete_json_raw(&self, req: JsonRequest) -> Result<serde_json::Value>;
    fn model(&self) -> &str;

    /// Whether the provider accepts a JSON Schema natively (e.g. OpenAI
    /// `response_format: json_schema`). `false` means the adapter renders the
    /// schema into the prompt and extracts JSON from the completion.
    fn supports_json_schema(&self) -> bool { true }

    /// Context window in tokens when known; the engine uses it to trim prompts.
    fn context_window(&self) -> Option<usize> { None }
}

/// Typed convenience layer. It cannot live on `Llm` itself: a generic method
/// would make the trait non-dyn-safe and `Box<dyn Llm>` impossible. The blanket
/// impl covers `dyn Llm` because it is `Llm + ?Sized`.
///
/// Structured-output repair is the adapter's responsibility: a provider that
/// cannot guarantee schema conformance (local models, `json_object`-only
/// endpoints) may feed the validation error back to the model a bounded number
/// of times. Only after that does a parse failure become `Error::Fatal` here.
#[async_trait]
pub trait LlmExt: Llm {
    async fn complete_json<T: DeserializeOwned>(&self, req: JsonRequest) -> Result<T>;
}

#[async_trait]
impl<T: Llm + ?Sized> LlmExt for T { /* raw JSON → T; parse failure → Error::Fatal */ }
```

**Local models are first-class adapters, not a fork of the engine.** Two shapes:

- **HTTP-compatible (default)**: the `llm` adapter is an OpenAI-compatible client; Ollama, vLLM, llama.cpp's server, SGLang, and LM Studio all expose compatible endpoints, so a local model usually needs configuration (`base_url`, `model`, optional `api_key`, timeout), not new code.
- **In-process (optional feature, e.g. `llm/local`)**: candle / ONNX Runtime / mistral.rs adapters implement `Llm` and `Embedder` directly. Inference is blocking GPU/CPU work, so it must run on `spawn_blocking` or a dedicated worker pool with bounded concurrency — never on the async runtime thread. Heavy model dependencies stay behind the feature flag so the default build stays light.

Local-specific concerns that stay inside adapters: timeout and retry policy (`Retryable` for transient failures), batch size for embeddings, dimension matching (`Embedder::dim`), grammar/guided decoding where available (GBNF, guided JSON), and schema rendering when `supports_json_schema()` is `false`. None of this changes the schema or the engine.

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
    // —— Bank & inbox (schema §5.1, four stages) ——
    async fn create_bank(&self, bank: Bank) -> Result<()>;
    async fn load_bank(&self, id: BankId) -> Result<Bank>;
    async fn create_source(&self, source: Source) -> Result<()>;                            // tx0
    async fn append_chunks(&self, src: SourceId, chunks: Vec<NewChunk>) -> Result<usize>;    // tx1; hash conflicts skipped
    async fn claim_chunk(&self, bank: BankId, now: Timestamp, stale_after: Duration)
        -> Result<Option<ClaimedChunk>>;                                                     // atomic pending→extracting claim
    async fn release_chunk(&self, chunk: ChunkId, now: Timestamp) -> Result<()>;              // on failure: back to pending (idempotent)
    async fn commit_chunk(&self, commit: ChunkCommit) -> Result<()>;                         // tx2: all products + status=done atomically

    // —— Read path (schema §5.4, four channels) ——
    async fn search_vector(&self, bank: BankId, q: &[f32], k: usize) -> Result<Vec<(FactId, f32)>>;
    async fn search_bm25(&self, bank: BankId, q: &str, k: usize) -> Result<Vec<(FactId, f32)>>;
    async fn neighbors(&self, bank: BankId, from: FactId) -> Result<Vec<Neighbor>>;          // fact_link + fact_entity-derived adjacency
    async fn facts_in_range(&self, bank: BankId, range: TimeRange) -> Result<Vec<FactId>>;
    async fn fetch_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>>;
    async fn increment_access(&self, ids: &[FactId]) -> Result<()>;

    // —— Entity / opinion / observation / maintenance ——
    async fn ensure_entity(&self, new: NewEntity) -> Result<(EntityId, bool)>;               // (id, created); UNIQUE conflict reuses
    async fn entity_candidates(&self, bank: BankId, mention: &str) -> Result<Vec<EntityCandidate>>;
    async fn insert_opinion(&self, op: NewOpinion) -> Result<()>;
    async fn insert_observations(&self, obs: Vec<NewObservation>) -> Result<()>;
    async fn observations_of(&self, bank: BankId, entity: EntityId, limit: usize) -> Result<Vec<Fact>>;
    async fn pending_work(&self, bank: BankId, now: Timestamp, stale_after: Duration) -> Result<WorkLists>;
    async fn update_embeddings(&self, embeddings: &[(FactId, Vec<f32>)]) -> Result<usize>;  // backfill write path
    async fn gc_observations(&self, bank: BankId, now: Timestamp, max_age: Duration,
                             per_entity_cap: usize) -> Result<usize>;
}
```

Deliberate contract trade-offs (all map to the schema's §1.3 criteria):

1. **`commit_chunk` is the only transaction entry**: atomicity is guaranteed by the trait method; transaction objects never leak, so engine does not need to understand each backend's transaction API.
2. **IDs and time are engine-supplied** (`AGENTS.md`): the store never generates business IDs and never reads the wall clock; every creation DTO carries `id` and `created_at`, and methods that need "now" take it as a parameter.
3. **`neighbors()` merges multiple sources inside the adapter**: `fact_link` rows + entity adjacency derived from `fact_entity` (the closure is never materialized) — criterion 3, "single source of truth". The result reports four kinds (`NeighborKind` includes `Entity`) because P2-3 applies a distinct µ per kind.
4. **Work lists are always derived**: `pending_work` returns sources awaiting chunking, chunks pending/stale-claim, NULL embeddings, and entities whose evidence changed (`entity.last_seen > latest observation row's created_at`). There is no task-queue table — criterion 2, "strategies live at runtime".
5. **No model/threshold fields**: embedding model, chunking parameters, σt/θs/top-k, µ(ℓ) all live in configuration — criterion 2. Embedding backfill therefore goes through `update_embeddings`: the engine owns the `Embedder`, the store only persists vectors.
6. **`search_vector` must always be available**: ANN is adapter-internal optimization (SQLite brute-force cosine; pgvector indexing can be added later on PG); the contract only requires "return top-k".

**Store shape (decided in P0-8): one trait.** The contract stays a single `Store` for the first version:

- Every method is object-safe, so `Box<dyn Store>` / `Arc<dyn Store>` injects cleanly and the engine has exactly one dependency to wire.
- Capability boundaries (read / write / maintenance) are *behavioral*, not *structural*: the same adapter owns the tables, and splitting would hand the engine two or three handles to the same connection pool without buying isolation.
- The composition would be a supertrait (`trait Store: StoreRead + StoreWrite + StoreMaintenance`) if it ever happens, so call sites typed as `&dyn Store` survive the split unchanged.

**Split trigger (recorded, not planned):** split when a consumer genuinely needs only a subset — e.g. a read-only serving replica (`search_*` / `fetch_facts` only), a maintenance worker running with narrower credentials (`pending_work` / `update_embeddings` / `gc_observations`), or a third-party adapter that can implement retrieval but not ingestion. The cut is then `StoreRead` (`load_bank`, `search_vector`, `search_bm25`, `neighbors`, `facts_in_range`, `fetch_facts`, `observations_of`, `entity_candidates`) / `StoreWrite` (`create_bank`, `create_source`, `append_chunks`, `claim_chunk`, `release_chunk`, `commit_chunk`, `ensure_entity`, `insert_opinion`, `insert_observations`, `update_embeddings`, `increment_access`) / `StoreMaintenance` (`pending_work`, `gc_observations`), with `commit_chunk` remaining the only transaction entry.

## 2. Pipeline Implementation Notes

### 2.1 retain (Inbox + Extraction)

```
ingest(payload):
  canonical text + turn spans (in memory)
  tx0: create_source (payload_text, status=received)
  chunking (session-alignment first / window fallback; spans still in memory)
  tx1: append_chunks (hash conflicts skipped) + source.status=chunked
  ── ack: return SourceId (durably queued; no model work has run yet)
worker loop (required for processing, one-shot `rescan` in tests):
  claim_chunk (pending→extracting; updated_at = claim token)
             → extract (Llm, A.1/A.5.1) → resolve (Store) → embed (Embedder)
             → build links (link module) → reinforce (Assess + confidence deltas)
             → commit_chunk (tx2, guarded by the claim token)
  // on failure: release_chunk, back to pending (or failed per retry policy)
rescan(): received sources → re-chunk; pending / stale-extracting chunks → re-run extraction
status: aggregate chunk state per source id (no progress table, no new column)
```

- All model calls (extraction/Assess/embedding) happen **outside** tx2; tx2 is pure database writes.
- `retain` is an **async ack** API: it returns the `SourceId` after tx1, so ack means "durably queued", not "done". Failures before the ack are returned as errors; failures after it are visible only through `retain_status` (`CompletedWithFailures` / `Failed`).
- Chunking happens **before** the ack on purpose: the normalized turn spans are still in memory, so acking after tx1 avoids losing session alignment on a crash between tx0 and tx1.
- A running worker is required to make progress; notifications are best-effort and the periodic sweep guarantees convergence after a lost wake-up.

### 2.2 recall (Four Channels → RRF → Rerank → Budget)

- `search_vector` / `search_bm25` run concurrently with graph expansion and temporal filtering (`try_join!`).
- RRF: `Σ 1/(60 + rank_i)`, aggregated by fact_id.
- Rerank: `Reranker` scores the top-N candidates (identity stub in the first version).
- Budget packing: greedily accumulate tokens in final order (pluggable estimator) and truncate on overflow; hit rows `access_count++`.

### 2.3 reflect (Answer + Opinion Formation + Reinforcement)

- recall evidence → assemble the system message (name n, background h, verbalized φ(Θ) + β-dependent wording modulation).
- One `Llm::complete_json_raw` call produces { answer text, candidate opinions[] } (A.2: first person, confidence, reasoning).
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
3. **Contract tests**: the same behavioral suite for `Store` runs against both SQLite and PG (test kit in the `store` crate); the PG half is deferred until P5-2.
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

1. ~~`Store` shape: single trait (current draft) vs capability sub-traits (read/write/maintenance) — settle in P0.~~ **Resolved in P0-8**: single trait; the split trigger and the proposed read/write/maintenance cut are recorded in §1.
2. ~~Whether retain's default API is synchronous or asynchronous (schema §7.B#7 recommends synchronous first).~~ **Resolved**: async ack — `retain` returns the `SourceId` after tx0+tx1; a worker processes chunks; progress is queried per source.
3. First implementations of reranker / temporal parser (identity stub + rule-based parser first; model-based versions deferred).
4. Tokenization for chunking and the budget estimator (do we reuse the embedding model's tokenizer?).
5. Evaluation integration timing and harness shape (P5).
6. Semantic follow-ups inherited from the schema doc: `§7.A#2` (nullable confidence on extraction-produced opinion rows vs an initial confidence).
