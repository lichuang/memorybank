# memorybank Implementation Steps (Phased Task List)

Based on `design.md` (implementation design) and `memory-schema.md` (finalized logical schema).

> **Status convention**: each task's `**Status**` line defaults to ❌; when complete, change it to ✅ (a date or commit hash is recommended).
> Current progress: **P0 12/12 · P1 19/20 · P2 10/10 · P3 0/8 · P4 1/9 · P5 0/10 (42/69 total)**

> **Scope note (storage)**: only SQLite is implemented. The PostgreSQL adapter and the dual-backend contract tests (P5-1/P5-2) are **deferred** until a second backend is actually needed; no `postgres` feature, module, or migration exists today. The `Store` contract remains backend-neutral.

## Common Conventions (apply to every step; not repeated below)

- **Async**: ports and orchestration are all `async` (`async-trait`); retain is an **async ack** API — it normalizes, persists the source + chunks (tx0/tx1), and returns the `SourceId`; extraction runs in the worker and progress is polled by source id.
- **Errors**: `engine::Error` (Retryable/Fatal/Config/NotFound/Conflict); workers retry only Retryable; tests may use `anyhow` helpers.
- **IDs**: UUIDv7, allocated in **engine before the transaction** (`FactId::new()`); `Store` only persists them — avoiding the backtracking dependency of "insert before you get an id".
- **Testing**: unit tests beside the module; integration tests in `crates/*/tests/`; all time goes through an injected `Clock`; an isolated in-memory SQLite database per test.
- **Configuration**: `engine::Config` (chunking/retrieval/linking/reinforcement/scheduling thresholds), serde-deserialized; engine reads no environment variables (the facade injects them).

---

# P0 · Foundation

## P0-1 Workspace Skeleton

**Status**: ✅ (2026-10-25)

- **Content**: root `Cargo.toml` (virtual workspace), `crates/{memorybank,engine,store,llm}`, `rustfmt.toml`, clippy lints, CI script.
- **Design**: workspace members; `llm/http` feature; `store` ships SQLite only.
- **Acceptance**: `cargo check/clippy/fmt --workspace` all green; `cargo tree` shows no reverse dependencies (engine does not depend on store/llm).
- **Tests**: CI runs `cargo test --workspace`.

## P0-2 ID and Time Primitives

**Status**: ✅ (2026-10-25)

- **Content**: 5 ID newtypes + a time alias.
- **Design**:
```rust
macro_rules! id_type { ($n:ident) => {
  #[derive(Clone,Copy,PartialEq,Eq,Hash,PartialOrd,Ord,Serialize,Deserialize)]
  #[serde(transparent)] pub struct $n(Uuid);
  impl $n { pub fn new() -> Self; pub fn parse(s:&str)->Result<Self>; pub fn as_uuid(&self)->Uuid; }
  impl Display for $n { ... }
}};
id_type!(BankId); id_type!(SourceId); id_type!(ChunkId); id_type!(FactId); id_type!(EntityId);
pub type Timestamp = chrono::DateTime<Utc>;
pub fn now_ts(clock:&dyn Clock) -> Timestamp
```
- **Acceptance**: serde round-trip; UUIDv7 ordering is creation-time ordered.
- **Tests**: round-trip; lexicographic monotonicity across successive `new()` calls.

## P0-3 Domain Enums and Structs

**Status**: ✅ (2026-10-25; extraction type split 2026-11-13)

- **Content**: all core enums/row structs.
- **Design**:
```rust
pub enum FactType { World, Experience, Opinion, Observation }
pub enum ExtractionFactType { World, Experience, Opinion }   // A.5.1 extraction contract
pub enum LinkType { Temporal, Semantic, Causal }
pub enum ChunkStatus { Pending, Extracting, Done, Failed }
pub enum CausalRelation { Causes, CausedBy, Enables, Prevents }   // kept as-is; relation expresses direction
pub enum EntityKind { Person, Organization, Location, Product, Concept, Other }
pub enum SourceStatus { Received, Chunked }

pub struct BankProfile { pub background: String, pub skepticism: u8, pub literalism: u8, pub empathy: u8, pub bias_strength: f32 }
impl BankProfile { pub fn validate(&self)->Result<()>; }   // 1..=5, 0..=1

pub struct Bank { id, name: String, profile: BankProfile, created_at, updated_at }
pub struct Source { id, bank_id, title: Option<String>, payload_text: String, status, created_at, updated_at }
pub struct Chunk { id, bank_id, source_id, chunk_index: u32, text: String, content_hash: String,
                   status: ChunkStatus, time_start: Option<Timestamp>, time_end: Option<Timestamp>, created_at, updated_at }
pub struct ClosedInterval { /* private */ start: Timestamp, end: Timestamp }   // new() rejects start > end
pub enum OccurrenceInterval { Closed(ClosedInterval), Open { start: Timestamp }, Unknown }
pub struct Fact { id, bank_id, fact_type: FactType, text: String, chunk_id: Option<ChunkId>,
                  embedding: Option<Vec<f32>>, occurred: OccurrenceInterval,
                  mentioned_at: Option<Timestamp>,
                  confidence: Option<f32>, access_count: u32, created_at, updated_at }
pub struct Entity { id, bank_id, canonical_name: String, kind: EntityKind,
                    mention_count: u32, first_seen/last_seen: Option<Timestamp>, created_at, updated_at }
pub struct FactEntity { bank_id, fact_id, entity_id, mention_text: Option<String> }
pub struct FactLink { id, bank_id, from_fact_id, to_fact_id, link_type: LinkType,
                      relation: Option<CausalRelation>, weight: f32 }
```
- **Acceptance**: one-to-one correspondence with `memory-schema.md` §3; invariants documented in type comments (observation ⇒ chunk_id=None; reflect opinion ⇒ chunk_id=None and confidence required).
- **Tests**: `BankProfile::validate` boundary matrix (0/1/5/6).

## P0-4 Structured Output DTOs (paper A.5.1/A.2/A.3)

**Status**: ✅ (2026-10-25; three-value extraction enum 2026-11-13)

- **Content**: LLM round-trip contracts + JSON Schema (schemars).
- **Design**:
```rust
pub enum ExtractionFactType { World, Experience, Opinion }   // no Observation: synthesized only
pub struct ExtractedFact { pub what: String, pub when: String, pub where_: String, pub who: String,
  pub why: String, pub fact_type: ExtractionFactType, pub confidence: Option<f32>,
  pub occurred_start: Option<Timestamp>, pub occurred_end: Option<Timestamp>,
  pub mentioned_at: Option<Timestamp>,
  pub entities: Vec<EntityMention>, pub causal_relations: Vec<CausalRef>, pub context: Option<String> }
pub struct EntityMention { pub text: String, pub kind: Option<EntityKind> }
pub struct CausalRef { pub target_fact_index: usize, pub relation: CausalRelation, pub strength: f32 }

pub struct FactExtractionResponse { pub facts: Vec<ExtractedFact> }
pub struct OpinionDraft { pub opinion: String, pub confidence: f32, pub reasoning: String }
pub struct ReflectResponse { pub answer: String, pub opinions: Vec<OpinionDraft> }
pub struct Assessment { pub verdict: AssessmentVerdict, pub reason: String } // Reinforce/Weaken/Contradict/Neutral
pub struct ObservationDraft { pub observation: String }
pub struct ObservationResponse { pub observations: Vec<ObservationDraft> }
```
- **Acceptance**: `schema_for!` snapshots align with the paper appendix field names (`where` via serde rename).
- **Tests**: fixture deserialization (including defaulted fields / wrong types).

## P0-5 Error System

**Status**: ✅ (2026-10-25)

- **Content**: `engine::Error` + retry classification.
- **Design**: `pub fn is_retryable(&self)->bool`; Store/Llm errors converted via `From`; `NotFound(&'static str, String)`.
- **Acceptance**: LLM timeout/DB busy → Retryable; schema parse failure → Fatal; profile out of range → Config.
- **Tests**: `is_retryable` matrix.

## P0-6 Model Ports

**Status**: ✅ (2026-10-25)

- **Content**: `Llm/Embedder/Reranker/TemporalParser/Clock`.
- **Design**:
```rust
pub struct JsonRequest { pub system: Option<String>, pub user: String,
  pub schema_name: &'static str, pub schema: serde_json::Value,
  pub temperature: f32, pub max_tokens: u32 }
#[async_trait] pub trait Llm: Send+Sync {                                  // dyn-safe
  async fn complete_json_raw(&self, req: JsonRequest) -> Result<serde_json::Value>;
  fn model(&self) -> &str;
  fn supports_json_schema(&self) -> bool { true }        // false → adapter renders the schema into the prompt
  fn context_window(&self) -> Option<usize> { None }     // used by the engine to trim prompts
}
#[async_trait] pub trait LlmExt: Llm {                                     // generic convenience layer
  async fn complete_json<T: DeserializeOwned>(&self, req: JsonRequest) -> Result<T>;
}
#[async_trait] impl<T: Llm + ?Sized> LlmExt for T { ... }                  // blanket: `dyn Llm` included
#[async_trait] pub trait Embedder: Send+Sync { fn model(&self)->&str; fn dim(&self)->usize;
  async fn embed(&self, texts:&[&str])->Result<Vec<Vec<f32>>>; }
#[async_trait] pub trait Reranker: Send+Sync { async fn rerank(&self, q:&str, docs:&[&str])->Result<Vec<f32>>; }
pub struct TimeRange { pub start: Timestamp, pub end: Timestamp }
pub trait TemporalParser: Send+Sync { fn parse(&self, q:&str, now: Timestamp)->Option<TimeRange>; }
pub trait Clock: Send+Sync { fn now(&self)->Timestamp; }
pub fn now_ts(clock: &dyn Clock) -> Timestamp            // truncated to millisecond precision
```
- **Acceptance**: `Box<dyn Llm>` compiles and can call both `complete_json_raw` and (via `LlmExt`) `complete_json`; the schema travels as request data (no generic schema bound).
- **Tests**: empty impl compiles + dyn invocation.

## P0-7 Store Port (Method Set)

**Status**: ✅ (2026-10-25)

- **Content**: the full storage contract + DTOs.
- **Design**:
```rust
pub struct NewEntity { pub id: EntityId, pub bank_id: BankId, pub canonical_name: String,
  pub entity_type: EntityKind, pub created_at: Timestamp }
pub struct NewChunk { pub id: ChunkId, pub bank_id: BankId, pub source_id: SourceId,
  pub chunk_index: u32, pub text: String, pub content_hash: String,
  pub time_start/time_end: Option<Timestamp>, pub created_at: Timestamp }
pub struct ClaimedChunk { pub chunk: Chunk }                        // status = Extracting
pub struct NewFact { pub id: FactId, pub bank_id: BankId, pub fact_type: FactType,
  pub text: String, pub chunk_id: Option<ChunkId>, pub embedding: Option<Vec<f32>>,
  pub occurred: OccurrenceInterval, pub mentioned_at: Option<Timestamp>,
  pub confidence: Option<f32>,  // required for opinion rows
  pub attrs: Option<serde_json::Value>, pub created_at: Timestamp }
pub struct NewBinding { pub bank_id: BankId, pub fact_id: FactId, pub entity_id: EntityId,
  pub mention_text: Option<String>, pub created_at: Timestamp }
pub struct NewLink { pub id: LinkId, pub bank_id: BankId, pub from: FactId, pub to: FactId,
  pub link_type: LinkType, pub relation: Option<CausalRelation>, pub weight: f32,
  pub created_at: Timestamp }
pub struct EntityStatUpdate { pub entity_id: EntityId, pub add: bool, pub at: Option<Timestamp>,
  pub updated_at: Timestamp }
pub struct OpinionConfidenceUpdate { pub opinion_id: FactId, pub new_confidence: f32,
  pub new_text: Option<String>, pub updated_at: Timestamp }
pub struct BackgroundUpdate { pub expected: String, pub background: String,
  pub updated_at: Timestamp }
pub struct ChunkCommit { pub chunk_id: ChunkId, pub claimed_at: Timestamp, pub completed_at: Timestamp,
  pub facts: Vec<NewFact>, pub entities: Vec<Entity>, pub bindings: Vec<NewBinding>,
  pub links: Vec<NewLink>, pub entity_stats: Vec<EntityStatUpdate>,
  pub opinion_updates: Vec<OpinionConfidenceUpdate>,
  pub background_update: Option<BackgroundUpdate> }
pub struct NewOpinion { pub id: FactId, pub bank_id: BankId, pub text: String,
  pub confidence: f32, pub entities: Vec<(EntityId, Option<String>)>,
  pub embedding: Option<Vec<f32>>, pub created_at: Timestamp }
pub struct NewObservation { pub fact_id: FactId, pub bank_id: BankId, pub text: String,
  pub embedding: Option<Vec<f32>>, pub entity_ids: Vec<EntityId>, pub created_at: Timestamp }
pub struct WorkLists { pub sources_received: Vec<SourceId>, pub chunks_pending: Vec<ChunkId>,
  pub chunks_stale: Vec<ChunkId>, pub facts_missing_embedding: Vec<FactId>,
  pub entities_stale_observation: Vec<EntityId> }
pub enum NeighborKind { Temporal, Semantic, Causal, Entity }
pub struct Neighbor { pub fact_id: FactId, pub kind: NeighborKind, pub weight: f32 }
pub struct EntityCandidate { pub entity: Entity, pub variants: Vec<String>,
  pub co_mentioned: Vec<EntityId>, pub last_mention_at: Option<Timestamp> }

pub enum FactScope { All, Types(Vec<FactType>) }

#[async_trait] pub trait Store: Send+Sync {
  // bank / inbox
  async fn create_bank(&self, bank:Bank)->Result<()>;
  async fn load_bank(&self, id:BankId)->Result<Bank>;
  async fn create_source(&self, source:Source)->Result<()>;
  async fn append_chunks(&self, src:SourceId, chunks:Vec<NewChunk>)->Result<usize>;  // inserted count
  async fn claim_chunk(&self, bank:BankId, now:Timestamp, stale_after:Duration)->Result<Option<ClaimedChunk>>;
  async fn release_chunk(&self, chunk:ChunkId, claimed_at:Timestamp, now:Timestamp)->Result<()>;  // token-guarded; pending/done/failed are no-ops
  async fn commit_chunk(&self, c:ChunkCommit)->Result<()>;
  // retrieval
  async fn search_vector(&self, bank:BankId, q:&[f32], k:usize, scope:&FactScope)->Result<Vec<(FactId,f32)>>;
  async fn search_bm25(&self, bank:BankId, q:&str, k:usize, scope:&FactScope)->Result<Vec<(FactId,f32)>>;
  async fn neighbors(&self, bank:BankId, from:FactId, scope:&FactScope)->Result<Vec<Neighbor>>;
  async fn facts_in_range(&self, bank:BankId, r:TimeRange, scope:&FactScope)->Result<Vec<FactId>>;
  async fn fetch_facts(&self, ids:&[FactId])->Result<Vec<Fact>>;
  async fn increment_access(&self, ids:&[FactId])->Result<()>;
  // entity / opinion / observation / maintenance
  async fn ensure_entity(&self, new:NewEntity)->Result<(EntityId,bool)>;   // (id, created)
  async fn entity_candidates(&self, bank:BankId, mention:&str, limit:usize)->Result<Vec<EntityCandidate>>;
  async fn merge_entities(&self, bank:BankId, from:EntityId, into:EntityId, now:Timestamp)->Result<()>;
  async fn insert_opinion(&self, o:NewOpinion)->Result<()>;
  async fn insert_observations(&self, o:Vec<NewObservation>)->Result<()>;
  async fn observations_of(&self, bank:BankId, entity:EntityId, limit:usize)->Result<Vec<Fact>>;
  async fn pending_work(&self, bank:BankId, now:Timestamp, stale_after:Duration)->Result<WorkLists>;
  async fn update_embeddings(&self, embeddings:&[(FactId,Vec<f32>)] )->Result<usize>;
  async fn gc_observations(&self, bank:BankId, now:Timestamp, max_age:Duration,
                           per_entity_cap:usize)->Result<usize>;
}
```
- **Key semantics** (documented in trait doc comments):
  - **IDs and time are engine-supplied** (`AGENTS.md`): the store never generates business IDs and never reads the wall clock; every creation DTO carries `id` and `created_at`.
  - `claim_chunk` claims atomically and reclaims stale `extracting` rows (`now - updated_at > stale_after`), refreshing `updated_at = now`; returns `None` when nothing is claimable.
  - `commit_chunk` is the only transaction entry for **extraction products**: facts / bindings / links / entity stats / opinion updates / the optional `bank.background` CAS / status→done (`completed_at`) commit together. If a listed entity hits `UNIQUE(bank_id, canonical_name)`, the store reuses the existing row and rewrites the bindings that referenced the new id. Maintenance operations (`gc_observations`, `merge_entities`) own separate transactions; they are not extraction products.
  - `merge_entities` moves every `fact_entity` binding from `from` to `into` (dropping source bindings whose `(fact_id, into)` already exists), recomputes `mention_count`/`first_seen`/`last_seen` from the surviving world/experience bindings, records the source id in `into.attrs.merged_from`, and deletes the source row. Both entities must belong to the bank; `from == into` is `Config`.
  - `append_chunks` skips any chunk whose `(bank_id, content_hash)` already exists and returns the inserted count.
  - Retrieval methods take a `FactScope` (`All` or a `Types` subset) so the target network semantics of paper eqs. (9)–(11) are expressed at the port; empty `Types` means no facts.
  - `neighbors` merges `fact_link` rows with adjacency derived from `fact_entity` (weight 1.0, `NeighborKind::Entity`); the pairwise closure is never materialized. Four kinds are reported because P2-3 applies a distinct `mu` per kind.
  - `insert_opinion` writes the opinion row plus its `fact_entity` bindings in one transaction; `insert_observations` appends observation rows plus bindings, never replacing an older batch.
  - `pending_work` derives all lists in-database; `update_embeddings` is the backfill write path (the engine owns the `Embedder`, so the store cannot re-embed by itself); `increment_access` batches counter bumps and leaves `updated_at` untouched.
  - `gc_observations` deletes observation rows older than `now - max_age`, then keeps the newest `per_entity_cap` rows per entity binding, in one transaction.
- **Acceptance**: the trait is object-safe; DTOs leak no backend types.
- **Tests**: compile-time object-safety test; DTO serde.

## P0-8 Store Shape Finalization

**Status**: ✅ (2026-10-25)

- **Content**: land the single-trait decision + a note on the splitting contingency (read/write/maintenance).
- **Decision** (recorded in `design/design.md` §1 "Store shape" and closing §6 item 1):
  - one object-safe `Store` trait; the engine holds a single `Box<dyn Store>` / `Arc<dyn Store>` handle;
  - capability boundaries are behavioral, not structural — one adapter owns the tables, so sub-traits would hand the engine several handles to the same pool without isolation;
  - if a split happens, it is a supertrait composition (`Store: StoreRead + StoreWrite + StoreMaintenance`), so existing `&dyn Store` call sites keep compiling.
  - **Split trigger**: a consumer needs only a subset — read-only serving replica, narrower-credential maintenance worker, or a third-party adapter that can serve retrieval but not ingestion. Proposed cut: `StoreRead` (load/search/neighbors/fetch/observations/candidates) / `StoreWrite` (bank, inbox, claim/commit, opinions, observations, embeddings, access) / `StoreMaintenance` (`pending_work`, `gc_observations`); `commit_chunk` stays the only transaction entry.
- **Acceptance**: `let s: Box<dyn Store>` compiles; the doc records the trigger for splitting (e.g. a read-only backend).
- **Tests**: dyn injection smoke test.

## P0-9 Migration 0001 (SQLite)

**Status**: ✅ (2026-10-26)

- **Content**: 8-table DDL + FTS + indexes + constraints.
- **Design**:
  - Tables follow the schema fields; `fact_vec` is a vec0 table (`fact_id`, `bank_id`, `embedding float[1536] distance_metric=cosine`) with a delete trigger; FTS5: `CREATE VIRTUAL TABLE fact_fts USING fts5(fact_id UNINDEXED, text, tokenize='trigram')` (maintained manually);
  - Unique: `chunk(bank_id, content_hash)`, `entity(bank_id, canonical_name)`, `fact_link(from_fact_id, to_fact_id, link_type, relation)`;
  - CHECK: enum domains; `(fact_type='opinion' AND chunk_id IS NULL) => confidence IS NOT NULL`;
  - Indexes: `fact(bank_id)`, `fact(chunk_id)`, `fact(bank_id, occurred_start)`, `source(bank_id,status)`, `fact_entity(bank_id, entity_id)`, `fact_entity(bank_id, fact_id)`, `fact_link(bank_id, from_fact_id)`, `fact_link(bank_id, to_fact_id)`.
- **Acceptance**: a line-by-line checklist against `memory-schema.md` §3 in the migration header.
- **Tests**: constraint violations (duplicate hash, duplicate canonical name, missing confidence); FTS insert/query.

## P0-10 Migration Runner + Test DB Helper

**Status**: ✅ (2026-10-26)

- **Content**: `store::migrate(&SqlitePool)`; `store::testing::memory_pool()`.
- **Acceptance**: migrations are idempotent (run twice); each test database is isolated.
- **Tests**: run migrate twice; basic concurrent connections.

## P0-11 Fakes

**Status**: ✅ (2026-10-26)

- **Content**: deterministic test doubles.
- **Design**:
```rust
pub struct FakeLlm { scripted: Mutex<HashMap<&'static str, VecDeque<serde_json::Value>>>, calls: Mutex<Vec<JsonRequest>> }
impl FakeLlm { pub fn push(&self, schema_name:&'static str, value:serde_json::Value); pub fn calls(&self)->Vec<JsonRequest>; }
pub struct HashEmbedder { dim: usize }        // character n-gram hash → normalized vector
pub struct IdentityReranker;
pub struct RuleTemporalParser;                 // "yesterday"/"last week"/ISO dates
pub struct FixedClock(pub Timestamp);
```
- **Acceptance**: same input, same output; `FakeLlm` allows asserting prompt content (for profile-verbalization tests).
- **Tests**: embedding determinism/dimension; FakeLlm returns Fatal when the queue is exhausted.

## P0-12 P0 Smoke Test

**Status**: ✅ (2026-10-26)

- **Content**: bank→source→chunks→claim/release integration test.
- **Scope note**: P0 has no `Store` implementation yet (it starts at P1-3), so the smoke test drives the same flow with runtime SQL against the migrated schema; the port-level version arrives with P1-3/P1-4. Covered: atomic claim, stale reclaim boundary, release back to `pending`, and rollback leaving no partial products.
- **Acceptance**: all green; `cargo doc` has no broken links (CI runs it with `RUSTDOCFLAGS=-D warnings`).
- **Tests**: `crates/store/tests/smoke.rs`.

---

# P1 · retain

## P1-1 Payload Normalization

**Status**: ✅ (2026-10-27; crash time recovery 2026-11-11)

- **Content**: turn caller input into canonical text + a turn view.
- **Design**:
```rust
pub struct RawTurn { pub speaker: String, pub text: String, pub at: Option<Timestamp>, pub session: Option<u32> }
pub enum RawPayload { Transcript { title: Option<String>, turns: Vec<RawTurn> }, Document { title: Option<String>, text: String } }
pub struct TurnSpan { pub byte_start: usize, pub byte_end: usize, pub at: Option<Timestamp>, pub session: Option<u32> }
pub struct NormalizedPayload { pub text: String, pub spans: Vec<TurnSpan> }
pub struct TimeHint { pub start: usize, pub end: usize, pub at: Timestamp }
impl RawPayload { pub fn title(&self) -> Option<&str> }   // retained for the source row
pub fn normalize(p: RawPayload) -> NormalizedPayload
pub fn recover_time_hints(text: &str) -> Vec<TimeHint>
// transcript line: "[<ts>] <speaker>: <text>\n"; the [<ts>] prefix is omitted when `at` is None
// <ts> = RFC3339 UTC with millisecond precision, e.g. 2024-06-09T12:00:00.000Z
```
- **Span contract**: `byte_start..byte_end` covers **only the turn text** (not `[ts] speaker: `, not the trailing `\n`), so `&normalized.text[start..end] == turn.text`. Sessions live in `TurnSpan.session` only (not in the text).
- **Crash recovery**: after a crash between tx0 and tx1, rescan keeps the window fallback but reparses line-start `[<rfc3339>] <speaker>: ` prefixes into `TimeHint`s, so chunk `time_start/time_end`, `mentioned_at`, and the fact time prefix survive. Session ids and exact turn boundaries are not recoverable (see `memory-schema.md` §5.1).
- **Acceptance**: Document passes through byte-for-byte (empty spans); Transcript keeps timestamps in text and sessions in spans; spans are invertible; no time is fabricated when `at` is `None`; recovery returns empty for untimed text and ignores malformed timestamp prefixes.
- **Tests**: empty turns, CJK (byte offsets, not chars), extremely long single turn, multi-line turn text, missing timestamp, title accessor, recovered time hints (valid/malformed/CJK).

## P1-2 Chunker

**Status**: ✅ (2026-10-27; recovery hints 2026-11-11; same-session merge 2026-11-12)

- **Content**: session alignment + window fallback + hashing.
- **Design**:
```rust
pub struct ChunkConfig { pub session_gap: Duration, pub max_tokens: usize, pub min_tokens: usize }
pub struct RawChunk { pub index: u32, pub text: String, pub content_hash: String,
                      pub time_start: Option<Timestamp>, pub time_end: Option<Timestamp> }
pub fn chunk(p: &NormalizedPayload, cfg: &ChunkConfig, est: &dyn TokenEstimator) -> Vec<RawChunk>
pub fn chunk_with_time_hints(text: &str, hints: &[TimeHint], cfg: &ChunkConfig,
                             est: &dyn TokenEstimator) -> Vec<RawChunk>
pub fn chunk_hash(text: &str) -> String   // whitespace-normalized + SHA-256 (lowercase hex)
// TokenEstimator + CharHeuristic live in engine::tokens (shared with P2-7)
```
- **Boundary rules**: chunks cut only at turn/line boundaries; a turn larger than `max_tokens` is split at UTF-8 char boundaries; a chunk below `min_tokens` merges into the previous chunk only when both belong to the **same session** and the result still fits `max_tokens` (session boundaries are never crossed); documents use pure windows; every chunk is a verbatim slice, whitespace-only pieces are dropped; `time_start/time_end` are the min/max turn times covered. The crash-recovery path (`chunk_with_time_hints`) keeps the window boundaries and attaches recovered timestamps to the window ranges they intersect.
- **Acceptance**: deterministic; never empty; oversized sessions split without crossing turn boundaries; whitespace-only hash differences collapse; recovered hints set `time_start/time_end` without changing window chunking; no hints preserve the existing document behavior.
- **Tests**: sessions by id and by gap, oversized session, oversized turn, CJK windows, same-session small-chunk merge, no merge across session ids or timestamp gaps, determinism, empty inputs, hash normalization, recovered hint bounds/intersection.

## P1-3 `create_source` (tx0)

**Status**: ✅ (2026-10-27)

- **Content**: first SQLite `Store` implementation + transaction boundary; bank creation/loading included so tests use real rows (option A).
- **Design**: `SqliteStore` wraps a migrated pool and implements `create_bank` / `load_bank` / `create_source`; the remaining trait methods return `Error::Fatal("not implemented yet")` until their own tasks land. Error mapping: `UNIQUE` → `Conflict`, `SQLITE_BUSY`/`SQLITE_LOCKED` → `Retryable`, everything else → `Fatal`; missing bank → `NotFound`; corrupt stored ids/timestamps/profile levels → `Fatal`.
- **Acceptance**: payload_text persisted, status=received; no half-writes before the crash point.
- **Tests**: read-back consistency; duplicated callers each get a new row (no meaningful external id); duplicate id/name → `Conflict`; missing bank → `NotFound`; `chunked` status round-trips.

## P1-4 `append_chunks` (tx1)

**Status**: ✅ (2026-10-28)

- **Content**: batch insert + conflict skip + source status advance.
- **Design**: `INSERT … ON CONFLICT(bank_id, content_hash) DO NOTHING` per row inside one transaction; the same transaction runs `UPDATE source SET status='chunked', updated_at=COALESCE(?, updated_at)` where the parameter is the **latest `created_at` in the batch** (option A; an empty batch keeps the old value). Chunks whose `source_id` differs from the argument are rejected with `Error::Config` before any write; inside the transaction the source's `bank_id` is read and every chunk must match it (`Error::Config` otherwise, schema §3.3). A missing source returns `NotFound` and nothing is written; any mid-batch failure rolls the whole transaction back.
- **Acceptance**: when every chunk conflicts, 0 rows are inserted but the source still becomes `chunked` (zero-chunk ledger row); a failed append leaves no chunk rows and leaves the source status untouched.
- **Tests**: same-source double ingest; identical content across sources (surviving row belongs to the first source); partial conflicts; empty batch; missing source; mismatched source id; mismatched bank id; verbatim field round-trip; mid-batch failure rollback.

## P1-5 `claim_chunk`

**Status**: ✅ (2026-10-28)

- **Content**: atomic claim + stale reclaim + monotonic claim token.
- **Design**:
```sql
UPDATE chunk SET status='extracting',
                 updated_at = MAX(?now, updated_at + 1)   -- strictly monotonic claim token
WHERE id = (SELECT id FROM chunk
            WHERE bank_id=?bank
              AND (status='pending'
                   OR (status='extracting' AND updated_at < ?stale_before))
            ORDER BY created_at, id LIMIT 1)
RETURNING *;
-- PG dialect would use GREATEST instead of MAX
```
- **Acceptance**: with two concurrent workers only one claims; stale `extracting` rows are reclaimed; each claim's `updated_at` is strictly greater than the previous one (even under `FixedClock`), so it can serve as the tx2 claim token.
- **Tests**: 10 concurrent claims, only 1 succeeds; oldest-first ordering; fresh claims not stolen; stale reclaim advances the token; same-millisecond claims get distinct tokens; `failed` rows never claimed; empty bank → `None`.

## P1-6 `release_chunk`

**Status**: ✅ (2026-10-29)

- **Content**: return a failed claim to `pending`.
- **Design**: `release_chunk(chunk, claimed_at, now)`; `UPDATE chunk SET status='pending', updated_at=? WHERE id=? AND status='extracting' AND updated_at=?claimed_at` (clock injected). 1 row → released. 0 rows → `Ok` when the row is already `pending` / `done` / `failed` (never resurrected); `Conflict` when it is still `extracting` with a different token (the claim was stolen); `NotFound` when missing.
- **Acceptance**: status=pending, updated_at refreshed; repeated failures keep the semantics; a second release is a no-op; `done` / `failed` rows stay untouched.
- **Tests**: after release the chunk can be claimed again; repeated release; done/failed untouched; missing chunk → `NotFound`.

## P1-7 Extraction Call

**Status**: ✅ (2026-10-29; schema-level observation rejection 2026-11-13)

- **Content**: A.1 prompt + schema + parse validation.
- **Design**:
```rust
pub struct ExtractionConfig { pub temperature: f32, pub max_tokens: u32,
  pub max_input_chars: usize, pub default_opinion_confidence: f32 }  // fallback 0.5
pub struct ExtractionReport { pub facts: Vec<ExtractedFact>, pub discarded: usize }
pub async fn extract_facts(llm:&dyn Llm, chunk:&Chunk, cfg:&ExtractionConfig) -> Result<ExtractionReport>
pub fn validate_extracted(f: &ExtractedFact) -> Option<&'static str>   // reason to discard
// prompt: prompts/extraction.md (A.1, adapted to the DTO), embedded with include_str!,
//         used as the system message; user = chunk.text; schema_name = "fact_extraction"
// schema: schema_for!(FactExtractionResponse), passed as request data
```
- **Validation scope**: `fact_type` is the three-value `ExtractionFactType`; `"observation"` (and any other unknown value) fails deserialization of the whole response (adapter repair may fix it; otherwise `Fatal`) instead of dropping a single fact after extraction. Per-fact checks are semantic: non-empty `what`; empty entity mentions are dropped. Opinion facts carry an initial `confidence` (0..1); a missing or out-of-range value falls back to `default_opinion_confidence` (0.5), and non-opinion facts never carry one.
- **Causal remapping**: model indices refer to the original `facts` array. After discarding invalid facts, surviving indices are remapped; references to dropped facts, out-of-range indices, and self-references are removed (logged).
- **Oversized input**: chunk text above `max_input_chars` is truncated on a UTF-8 char boundary with a `tracing::warn` (0 disables the safety net; chunking should already bound the size).
- **Acceptance**: invalid single facts are discarded and counted; causal references stay valid after filtering; opinion confidence is always present (fallback applied) and non-opinion facts have none; call-level retry only for `Retryable`.
- **Tests**: FakeLlm returns 3 facts with 1 bad; causal remapping across a dropped fact; self-reference removal; observation rejected at deserialization; empty result; UTF-8 truncation; schema mismatch → `Fatal`; `Retryable` propagation; opinion confidence kept, missing/invalid falls back, non-opinion confidence cleared.

## P1-8 Time and Augmented Text

**Status**: ✅ (2026-10-29)

- **Content**: time normalization + human-readable prefix + embedding-input assembly.
- **Design**:
```rust
pub fn normalize_time(extracted:&mut ExtractedFact, chunk:&Chunk)
// mentioned_at ← chunk midpoint (or its only bound); never clock.now()
// occurred_* are never derived from the chunk (conversation window ≠ event interval)
pub fn human_readable(ts: Timestamp) -> String            // "Sunday, June 9, 2024" (no padding)
pub fn augmented_text(narrative:&str, ts: Option<Timestamp>) -> String  // "{human}. {narrative}" / narrative
pub fn fact_time(extracted:&ExtractedFact) -> Option<String> // occurrence date/range, fallback mentioned_at
pub fn fact_text(extracted:&ExtractedFact) -> String      // what + fact_time prefix + missing dimensions
pub fn fact_attrs(extracted:&ExtractedFact) -> Option<serde_json::Value>  // structured when/where/who/why
```
- **Composition**: `fact.text` = `fact_time` prefix + `what`, plus a deterministic `(when: …; where: …; who: …; why: …)` suffix for any non-empty dimension not already present in the narrative; `fact.attrs` stores the four structured fields. `fact_time` is occurrence-first: a single date when `occurred_start == occurred_end`, `From <start> to <end>` for an interval, a one-sided occurrence for a single bound, and `mentioned_at` only when the occurrence is unknown. The prompt still asks `what` to weave the dimensions, but the fallback prevents permanent loss when the model does not. `fact_text` remains both `fact.text` and the embedding/reranker input (one string across all paths, schema §7.A#1).
- **Validation order**: `extract_facts` calls `normalize_time` before `validate_extracted`. `normalize_time` only fills `mentioned_at`; occurrence bounds stay as the model reported them. An interval with `end < start`, or an end without a start, discards the fact (causal references to it are remapped away); a start-only interval is a valid open interval, so future events are kept.
- **Typed interval**: `Fact` and `NewFact` carry `occurred: OccurrenceInterval` instead of two independent options. `OccurrenceInterval::from_parts` rejects reversed and end-only bounds, so an invalid interval cannot exist in a fact; the store skips a corrupt legacy row individually (with a warning) instead of failing the whole read.
- **No fabrication**: occurrence bounds are never derived from the chunk window; without them only `mentioned_at` is filled. `occurred_end` is never synthesized from `occurred_start`.
- **Acceptance**: never fabricate time (None stays None); stored text == embedding input (the same string); the prefix uses the occurrence interval when known and falls back to the mention time; occurrence is never fabricated from the chunk; prefix format `"<weekday>, <month> <day>, <year>. "` (ranges use `"From … to …. "`); dimensions missing from `what` are appended and copied to `attrs`.
- **Tests**: only the mention time is filled from chunk bounds; occurrence stays `None` when the model gave none; a future start-only interval is kept as open; end-only and reversed intervals are discarded; user-supplied times untouched; no chunk times → all times `None`; human-readable format (weekday correctness); prefix only when a time exists; occurrence preferred over mention; occurrence ranges; mention fallback; `fact_text` appends only missing dimensions and does not duplicate woven ones; `fact_attrs` keeps the structured fields.

## P1-9 Entity Resolution ρ

**Status**: ✅ (2026-10-29; tx2 creation boundary fixed 2026-11-10; config validation 2026-11-16)

- **Content**: exact hit → three-feature fuzzy match → a new-entity proposal that tx2 inserts.
- **Design**:
```rust
pub struct ResolveWeights { pub alpha: f32, pub beta: f32, pub gamma: f32,
  pub threshold: f32, pub temporal_scale: Duration, pub candidate_limit: usize }
// defaults: 0.6 / 0.2 / 0.2 / 0.75 / 30 days / 500
pub struct MentionContext { pub co_mentioned: Vec<EntityId>, pub at: Option<Timestamp> }
pub struct ResolvedEntity { pub id: EntityId, pub new_entity: Option<NewEntity> }
impl ResolveWeights { pub fn validate(&self)->Result<()> }
pub async fn resolve(store:&dyn Store, bank:BankId, mention:&EntityMention,
                     ctx:&MentionContext, w:&ResolveWeights, now:Timestamp,
                     pending:&[NewEntity]) -> Result<ResolvedEntity>
pub fn normalize_name(s:&str)->String                 // lowercase, trim, collapse whitespace
pub fn string_sim(a:&str,b:&str)->f32                 // 1 - levenshtein / max_len (chars)
pub fn cooccurrence_sim(cand:&EntityCandidate, ctx:&MentionContext)->f32  // overlap coefficient
pub fn temporal_sim(cand:&EntityCandidate, at:Option<Timestamp>, scale:Duration)->f32 // exp(-Δt/scale)
```
- **Candidate shortlist** (`Store::entity_candidates(bank, mention, limit)`): exact canonical-name and `mention_text` variant matches first, then the bank's top entities by `mention_count`, capped at `limit`. `EntityCandidate.co_mentioned` is a bounded sample of entities sharing a fact with the candidate (fact ids are UUIDv7, so recent-first). Fuzzy scoring runs on this shortlist; `sim_str` takes the best score over the canonical name and all variants. Proposals from earlier mentions in the same chunk (`pending`) are appended as zero-evidence candidates, so exact/fuzzy matching cannot split one name into several proposals.
- **Scoring**: weighted sum over **available** features, divided by the sum of their weights (a missing co-occurrence/temporal signal does not depress the score). Empty mention after normalization → `Error::Config`.
- **Validation**: `ResolveWeights::validate` requires `alpha > 0`, finite non-negative `beta`/`gamma`, `threshold ∈ [0, 1]`, positive `temporal_scale`, and `candidate_limit > 0`; `resolve` calls it before scoring. The threshold remains a deliberate deviation from paper eq. (2) (pure argmax): without it, a genuinely new mention would be forced onto the highest-scoring entity.
- **Creation**: below the threshold → `ResolvedEntity { id, new_entity: Some(NewEntity) }`; **nothing is written**. `entity_type = mention.kind.unwrap_or(Other)`; `created_at = now` (the only clock use). The worker puts proposals into `ChunkCommit.entities`; tx2 inserts them with `ON CONFLICT(bank_id, canonical_name) DO NOTHING`, and a conflict remaps the proposer's bindings/statistics to the winning row.
- **Acceptance**: `resolve` performs no writes; entity rows are created only in tx2; the same mention always resolves to the same entity; concurrent creation of one name converges through the tx2 conflict path; empty mention rejected.
- **Tests**: exact/fuzzy-typo/threshold/new-entity proposal paths; pending exact/fuzzy reuse; variant and co-mention candidates; candidate ordering and limit; empty mention; no entity row before tx2; concurrent resolution of an existing name is stable; invalid weight/threshold/limit configurations are rejected.

## P1-10 Entity Statistics

**Status**: ✅ (2026-10-30)

- **Content**: apply `EntityStatUpdate` inside the commit.
- **Design**: a store-internal
  `apply_entity_stats(&mut Transaction, bank, &[EntityStatUpdate])` stage, called by
  `commit_chunk` (P1-16) after the binding writes, inside the same transaction.
  `add=true` → `mention_count += 1`; `add=false` → `mention_count -= 1`, floored at 0
  (`CHECK (mention_count >= 0)`). On inserts, `at=Some` moves `first_seen`/`last_seen`
  to the min/max; `at=None` leaves both untouched. Deletions never move the seen
  bounds, even when they carry a time (the bounds are history, not a live recount).
  A zero-row update (unknown entity or an entity from another bank) is
  `Error::Fatal`: the commit references data it does not own, and tx2 must roll back.
- **Acceptance**: statistics stay consistent with the binding-row count (tests assert
  `COUNT(* FROM fact_entity)`); a failed update aborts the caller's transaction with
  no partial rows; `first_seen`/`last_seen` follow add-time only.
- **Tests**: count/bounds on add; earlier/later adds move the bounds; removal floors
  at zero and never rewinds the bounds; `None` time leaves the bounds; unknown entity
  and wrong bank are fatal and roll back; count equals the binding count.

## P1-11 Embedding Generation

**Status**: ✅ (2026-10-30)

- **Content**: batch embedding + dimension validation + failure degradation.
- **Design**: `pub async fn embed_facts(emb:&dyn Embedder, texts:&[String], batch:usize)
  -> Result<Vec<TextEmbedding>>` with
  `pub enum TextEmbedding { Vector(Vec<f32>), Failed }`; `TextEmbedding::into_option()`
  converts to the storage shape (`None` = backfill later). Splits inputs with
  `chunks(batch)` and preserves order. Provider `Retryable`/`Fatal` errors mark that
  batch `Failed` and log a warning; later batches still run, and the rows are backfilled
  from `facts_missing_embedding`, so a model outage never blocks tx2. A provider
  `Config` error is returned immediately — silently emitting null embeddings would hide
  a broken setup. A vector whose length differs from `Embedder::dim()` is
  `Error::Config` (writing it would poison vector search); a provider returning a
  different number of vectors than the batch size is `Error::Fatal` (outputs cannot be
  aligned with the texts). `batch == 0` → `Config`; empty input skips the provider.
- **Acceptance**: dimension mismatch is a Config error; failure does not block tx2;
  output length and order match the input.
- **Tests**: batch splitting and order preservation; single-batch `Retryable` failure →
  `Failed` slots while later batches continue; `Fatal` also degrades; provider `Config`
  aborts before later batches; storage conversion; dimension mismatch; vector-count
  mismatch; zero batch; empty input.

## P1-12 Temporal Link Construction

**Status**: ✅ (2026-11-03)

- **Content**: interval proximity → two rows.
- **Design**:
```rust
pub fn temporal_weight(gap: Duration, sigma: Duration) -> f32   // paper §4.1.4 eq. (4); σ=0 → step
pub async fn build_temporal_links(store:&dyn Store, bank:BankId, new_facts:&[Fact],
                                  cfg:&LinkConfig, now:Timestamp) -> Result<Vec<NewLink>>
pub struct LinkConfig { pub temporal_window: Duration /* 24h */,
                        pub temporal_sigma: Duration  /* 6h */,
                        pub temporal_top_k: usize     /* 5 */,
                        pub semantic_top_k: usize     /* 5, added in P1-13 */,
                        pub semantic_threshold: f32   /* 0.8, added in P1-13 */ }
```
  Uses **occurrence intervals** (`occurred_start..occurred_end`), as clarified in
  `memory-schema.md` §5.5: `gap = 0` for overlapping or touching intervals, otherwise the
  empty span between them; both bounds are required (an open interval creates no edge).
  Candidate windows around the new facts are merged, then `Store::facts_in_range`
  (occurrence intersection; `occurred_end IS NULL` = open interval) and
  `Store::fetch_facts` load the existing side; new-new pairs are matched in memory.
  Each new fact keeps its `temporal_top_k` closest candidates, pairs are deduplicated by
  canonical key, and each surviving pair yields `from→to` plus `to→from` with equal weight
  and `created_at = now` (the only clock use).
  **Coverage note**: occurrence bounds are never derived from the chunk window (P1-8), so
  only facts whose time the model actually resolved participate in temporal links and range
  queries. Ongoing facts (`Open`, e.g. "working at Google since May 2023") currently get no
  temporal edges; a future extension may treat them as `[start, +∞)`.
- **Acceptance**: reciprocal pairs always complete (canonical-key count is 2); pruning
  leaves no one-sided rows; rows without a complete time interval create no edges.
- **Tests**: weight formula (including σ=0); interval gap; three-fact time window;
  per-fact budget; new-new dedup; incomplete intervals; inclusive window boundary;
  cross-bank isolation; deterministic selection; `facts_in_range` intersections and
  `fetch_facts` round-trip / corrupt-BLOB handling.

## P1-13 Semantic Link Construction

**Status**: ✅ (2026-11-05)

- **Content**: directed top-k.
- **Design**:
```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32   // paper §4.1.4 eq. (5); 0 for zero norms
pub async fn build_semantic_links(store:&dyn Store, bank:BankId, new_facts:&[Fact],
                                  cfg:&LinkConfig, now:Timestamp) -> Result<Vec<NewLink>>
```
  For every new fact with an embedding, `Store::search_vector` (SQLite: the `fact_vec` vec0
  index created by migration 0001, fixed 1536 dimensions, cosine distance, triggers mirror
  the `fact_vec` table, which is the sole vector storage) supplies committed candidates; a dimension mismatch is `Config`; the other
  new facts of the batch are compared in memory because they are not committed yet. Candidates
  are deduplicated by id, filtered by `cos >= semantic_threshold`, sorted by `(score desc, id
  asc)`, and truncated to `semantic_top_k`. Each selected candidate yields exactly one directed
  row (`from = new`, `to = candidate`, `weight = cos` clamped to `[0, 1]`); no reverse row is
  invented. Facts without embeddings produce no edges (backfilled later).
  `semantic_threshold` must be in `[0, 1]`; `semantic_top_k = 0` disables the builder.
- **Acceptance**: directed single rows; no B→A unless B also selects A; within budget.
- **Tests**: cosine unit tests; vector-search ranking/limit/tie-break/dimension mismatch/empty
  bank; similar vs weak candidates; top-k budget; new-new reciprocal selection; self exclusion;
  missing embeddings; out-of-range threshold; deterministic selection.

## P1-14 Causal Link Construction

**Status**: ✅ (2026-11-05)

- **Content**: map extraction output.
- **Design**:
```rust
pub fn causal_links(bank:BankId, extracted:&[ExtractedFact], ids:&[FactId],
                    now:Timestamp) -> Vec<NewLink>
```
  `ids[i]` is the persisted id of `extracted[i]` (the extraction pass already remapped causal
  indices after dropping invalid facts). `target_fact_index → ids[target]`; one directed row
  per `(from, to, relation)` with `link_type=Causal`, `weight=1.0` (the model-reported
  `strength` never enters the weight), and `created_at=now`. The extracted relation is kept
  as-is: no endpoint swapping and no relation narrowing (`caused_by` means the target is the
  cause). Out-of-range targets, missing source ids, and self-references are dropped.
- **Acceptance**: out-of-range indices dropped; duplicate triples deduplicated; strength does
  not enter weight (paper w=1.0); direction follows the extracted relation.
- **Tests**: one-way causality; `caused_by` keeps direction; chained causality; out-of-range
  and self references; duplicate triples vs distinct relations; missing ids / empty input.

## P1-15 Opinion Reinforcement

**Status**: ❌ (on hold — opinion update semantics under review)

- **Hold reason**: text rewrite and multi-fact aggregation rules are unresolved; the commit path (P1-16) still accepts `opinion_updates`, so this can be wired later without blocking the pipeline.
- **Content**: candidates → Assess → confidence update.
- **Design**:
```rust
pub struct ReinforcementConfig { pub alpha: f32, pub sim_threshold: f32, pub max_candidates: usize }
pub async fn reinforce(store:&dyn Store, llm:&dyn Llm, bank:BankId,
                       new_facts:&[Fact], cfg:&ReinforcementConfig)
  -> Result<Vec<OpinionConfidenceUpdate>>
// candidates: entity overlap ∪ sim(v_o,v_f)>θ (search_vector with FactScope::Types([Opinion]))
// Assess schema: {"verdict":"reinforce|weaken|contradict|neutral","reason":"…"}
// c' = min(c+α,1) / max(c-α,0) / max(c-2α,0) / c
```
- **Acceptance**: four-verdict rules unit-tested; neutral produces no update; contradict may rewrite text; everything is atomic with tx2.
- **Tests**: FakeLlm four verdicts; both candidate paths (entity overlap / vector recall); α boundaries (0/1).

## P1-16 `commit_chunk` (tx2)

**Status**: ✅ (2026-11-06)

- **Content**: commit every product in one transaction, guarded by the claim token.
- **Design**: the bank is resolved from the chunk row (the DTO carries none). Insert order:
  background CAS (when `background_update` is present) → `entity` → `fact` (+ `fact_vec` when an
  embedding is present) → `fact_entity` → `fact_link` → `fact_fts` → entity statistics → opinion
  updates. In retain, entity rows are **created here and only here**: `resolve` returns proposals,
  and a failed chunk must leave no entity orphans. The background CAS is
  `UPDATE bank SET background=?, updated_at=? WHERE id=? AND background=?`; 0 rows means another
  chunk merged first, so the transaction rolls back with `Error::Conflict("background", …)` and
  the worker re-merges. Entity rows use
  `ON CONFLICT(bank_id, canonical_name) DO NOTHING`; when a canonical-name conflict reuses an
  existing id, the commit rewrites the ids in its bindings and statistic deltas before writing
  them. Finally, flip `chunk.status='done'` with
  `WHERE id=? AND status='extracting' AND updated_at=?claimed_at`; 0 rows updated ⇒ the claim
  was stolen ⇒ `Error::Conflict` and the whole transaction rolls back. Any failure rolls back
  explicitly.
- **Acceptance**: no residual rows after failure; after success, facts/edges/statistics are
  consistent; a second committer with an outdated `claimed_at` is rejected and leaves no rows.
- **Tests**: full product commit; stale-token rejection with no writes; replay rejection;
  injected duplicate-fact failure with full rollback; canonical-name conflict rewrites bindings
  and statistics.

## P1-17 retain API (Async Ack) + Status Query

**Status**: ✅ (2026-11-06)

- **Content**: facade ingest entry point + per-source progress query.
- **Design**:
```rust
impl MemoryBank {
  pub fn new(store:Arc<dyn Store>, clock:Arc<dyn Clock>, cfg:RetainConfig) -> Self;
  pub async fn retain(&self, bank:BankId, payload:RawPayload) -> Result<SourceId>;
  pub async fn retain_status(&self, bank:BankId, source:SourceId) -> Result<RetainStatus>;
}
pub struct RetainConfig { pub chunk: ChunkConfig, pub stale_after: Duration } // 10 min
pub struct RetainStatus { pub state: RetainState, pub chunks_total/pending/extracting/done/failed: usize,
                          pub facts_written: usize }
pub enum RetainState { Received, Queued, Running, Stalled, Completed,
                       CompletedWithFailures, Failed }
```
  `retain` normalizes → tx0 (`source.received`) → chunks (spans still in memory) → tx1
  (`append_chunks` + `source.chunked`) → returns `SourceId`. The ack means "durably queued";
  no model call runs before it. An empty chunk batch still flips the source to `chunked`.
- **Status derivation** (no progress table, no new column): `Store::source_progress` aggregates
  `chunk` rows by `(bank_id, source_id)` and counts committed facts. `Received` when the source
  is `received` with zero chunks; total=0 on a `chunked` source (all hashes collided) →
  `Completed`; an `extracting` row older than `stale_after` → `Stalled`; otherwise extracting →
  `Running`, pending → `Queued`, all done → `Completed`, all failed → `Failed`, mixed done +
  failed → `CompletedWithFailures`. Unknown source → `NotFound`.
- **Index**: `chunk(bank_id, source_id)` is part of the initial migration.
- **Acceptance**: `retain` returns with `source.status=chunked` and all chunk rows visible but
  zero facts; `retain_status` matches chunk states at every boundary; without a worker the state
  stays `Queued/Running`; re-ingest of identical material yields a zero-chunk source in
  `Completed`.
- **Tests**: ack point (source + chunks, no facts); claim → `Running`; stale claim → `Stalled`;
  done → `Completed`; zero-chunk re-ingest; unknown source; `Received`; mixed done/failed →
  `CompletedWithFailures`; all failed → `Failed`; `facts_written` count.

## P1-18 worker + rescan

**Status**: ✅ (2026-11-07)

- **Content**: background processing driven by derived work lists.
- **Dependency upgrade (prerequisite)**: ✅ (2026-11-07) sqlx 0.9, Rust 1.94, `libsqlite3-sys` 0.37; adapted the test SQL helper to `SqlSafeStr`; tests, clippy, and docs green.
- **Design**:
```rust
pub struct WorkerConfig { pub concurrency: usize, pub poll_interval: Duration,
  pub stale_after: Duration, pub max_attempts: u32, pub retry_delay: Duration }
pub struct PipelineConfig { pub extraction: ExtractionConfig, pub links: LinkConfig,
  pub resolve: ResolveWeights, pub embedding_batch: usize, pub background: BackgroundConfig }
impl MemoryBank {
  pub fn with_models(llm: Arc<dyn Llm>, embedder: Arc<dyn Embedder>) -> Self;
  pub fn with_pipeline(cfg: PipelineConfig) -> Self;
  pub fn with_worker_config(cfg: WorkerConfig) -> Self;
  pub fn spawn_worker(&self, cfg: WorkerConfig) -> JoinHandle<()>;   // rescan loop + sleep
  pub async fn rescan(&self, bank: BankId) -> Result<RescanReport>;  // one deterministic pass
}
pub struct RescanReport { pub sources_chunked: usize, pub chunks_processed: usize,
                          pub failed: usize }
```
  `rescan` reads `pending_work`, re-chunks `received` sources from `payload_text` (window
  fallback; spans are not persisted, but `recover_time_hints` reparses the canonical `[<ts>]`
  prefixes into chunk time bounds), then loops `claim_chunk` over pending/stale rows. Each
  chunk runs `extract_facts → resolve → embed_facts → temporal/semantic links → causal_links →
  load bank + merge_background (LLM) → commit_chunk` (opinion updates stay empty while P1-15 is
  on hold). The merge result rides in `ChunkCommit::background_update`, so A3 commits atomically
  with the chunk; a stale background CAS is retried with a fresh read, and an exhausted retry
  budget returns `Retryable`. Failures return the chunk
  to `pending`; attempts are counted in memory and, at `max_attempts`, the chunk is
  dead-lettered via `fail_chunk`. A `Conflict` from the claim-token CAS means another worker
  owns the chunk and is left untouched. `concurrency` is reserved (the pass is sequential);
  `retry_delay` only backs off the spawned loop, so `rescan` stays sleep-free and deterministic.
  `retain` sends a best-effort `Notify` wake-up; the periodic sweep covers lost notifications.
- **Store additions**: `list_banks`, `load_source`, `fail_chunk`, `fetch_entities`, plus the
  derived `pending_work` implementation (missing embeddings = facts without a `fact_vec` row).
- **Acceptance**: after `retain` + one `rescan` pass every chunk reaches `done`; stale
  `extracting` rows are reclaimed; `failed` rows stay put; a worker started with a long
  interval still converges via the periodic sweep.
- **Tests**: retain + rescan produces facts, vectors, bindings, statistics, and FTS rows;
  failed extraction is dead-lettered and not retried; stale claims are reclaimed; `received`
  sources are re-chunked; the spawned worker drains a queued source; missing models surface as
  failed chunks.

## P1-19 retain Integration Tests

**Status**: ✅ (2026-11-17)

- **Content**: end-to-end matrix for the async contract.
- **Acceptance/Tests**:
  - ack semantics: `retain` returns `SourceId` with chunks present and **zero facts**; `retain_status = Queued`;
  - after one `rescan` pass (no sleeps): `retain_status = Completed` and the expected facts/entities/links exist;
  - re-ingest idempotency (zero new facts; zero-chunk source in `Completed`); incremental re-ingest (only new chunks processed);
  - interrupted extraction → rescan recovers; tx2 failure → no partial state;
  - stale claim reclaimed without double-committing (claim-token CAS);
  - two-row invariant (SQL asserts count=2); entity statistics match bindings;
  - `CompletedWithFailures` when a chunk is dead-lettered;
  - background snippets update `bank.background` atomically with the chunk; a stale CAS leaves
    no facts and no background change;
  - a failure after resolution (e.g. background merge) leaves zero `entity` rows: proposals
    never write before tx2;
  - a crash between tx0 and tx1 recovers chunk `time_start/time_end` and fact `mentioned_at`
    from the canonical `[<ts>]` prefixes during rescan.
- **Coverage map**: ack/status derivation in `crates/memorybank/tests/retain_api.rs`;
  the worker pipeline, incremental re-ingest, temporal two-row/statistics consistency,
  `CompletedWithFailures`, background atomicity, no-orphan resolution, tx2 CAS rollback,
  and crash time recovery in `crates/memorybank/tests/worker.rs`; claim-token replay and
  transaction rollback at the store boundary in `crates/store/tests/commit_chunk.rs`.

## P1-20 Background Merging

**Status**: ✅ (2026-11-10)

- **Content**: retain step 5 (paper §5.6/§6.1): `h' = MergeLLM(h, h_new)` — extract durable
  first-person background snippets in A.1, merge them with the bank profile, and persist the
  result atomically with the chunk.
- **Design**:
  - A.1 response gains `background_snippets: Vec<String>` (`#[serde(default)]`); the extraction
    prompt asks for durable first-person biographical facts and an empty list when none exist.
    `ExtractionReport` carries the normalized snippets (trimmed, empty/deduped, order preserved).
  - `retain/background.rs`:
    ```rust
    pub struct BackgroundConfig { pub temperature:f32, pub max_tokens:u32,
                                  pub max_chars:usize, pub max_attempts:u8 }
    pub async fn merge_background(llm:&dyn Llm, current:&str, snippets:&[String],
                                  cfg:&BackgroundConfig) -> Result<String>;
    ```
    A dedicated prompt (paper §5.6 rules: new information wins direct conflicts, append
    non-conflicting details, keep first person, stay concise) returns
    `BackgroundMergeResponse { background: String }`. Empty output or output over `max_chars`
    is `Fatal`.
  - `Store::ChunkCommit` gains `background_update: Option<BackgroundUpdate>` with
    `expected`/`background`/`updated_at`. `commit_chunk_tx` applies the CAS **before** products;
    0 rows ⇒ `Error::Conflict("background", …)` and the whole tx rolls back. There is no
    standalone `update_bank_background` method: A3 must share tx2 with the chunk, and a second
    transaction entry would break that atomicity.
  - Worker: after links, when snippets exist, `load_bank` → `merge_background` (outside tx) →
    `commit_chunk`; no snippets ⇒ `background_update=None` and no LLM call. A background CAS
    conflict reloads, re-merges, and retries up to `max_attempts`; exhaustion returns
    `Retryable` so the chunk returns to `pending`. A chunk-claim conflict is never retried as a
    background conflict.
- **Acceptance**: snippets produce a coherent first-person `bank.background` in the same
  transaction as the chunk; empty snippets leave it untouched; a stale CAS writes neither facts
  nor background; merge failure fails the chunk before tx2 (no partial products).
- **Tests**: extraction snippet normalization/schema default; merge prompt and conflict rules
  via `FakeLlm`; CAS apply + stale rollback; worker update + no-snippet skip + retry budget;
  `max_chars` validation.

---

# P2 · recall

## P2-1 Semantic Channel

**Status**: ✅ (2026-11-17)

- **Design**: the store search already exists (vec0 exact KNN, `FactScope` scoping). This task adds `engine::recall::semantic`:
```rust
pub struct SemanticConfig { pub top_k: usize, pub min_cos: f32 }  // defaults: 50 / 0.0
impl SemanticConfig { pub fn validate(&self)->Result<()> }        // min_cos finite in [-1,1]
pub async fn semantic_channel(store:&dyn Store, bank:BankId, query:&[f32],
                              cfg:&SemanticConfig, scope:&FactScope)
  -> Result<Vec<(FactId,f32)>>;
```
  It calls `Store::search_vector(bank, query, cfg.top_k, scope)`, drops scores below
  `min_cos` (inclusive), preserves the store's ranked order, and truncates to `top_k`.
  `top_k = 0` is a valid disabled state and returns empty without a store call. No
  brute-force reimplementation and no fact loading (P2-8 assembles products later).
- **Acceptance**: respects the target `FactScope`; inclusive `min_cos` filtering; rows
  without embeddings are skipped; empty bank / disabled channel return empty.
- **Tests**: known-vector ordering, top-k, exact-threshold inclusion, scope isolation,
  missing embeddings, empty bank, zero top-k, config validation.

## P2-2 BM25 Channel

**Status**: ✅ (2026-11-17)

- **Design**: `engine::recall::bm25` wraps `Store::search_bm25`:
```rust
pub struct Bm25Config { pub top_k: usize }  // default 50; 0 disables
pub async fn bm25_channel(store:&dyn Store, bank:BankId, query:&str,
                          cfg:&Bm25Config, scope:&FactScope)
  -> Result<Vec<(FactId,f32)>>;
```
  SQLite canonicalizes query terms (alphanumeric runs of at least 3 characters,
  the trigram minimum), wraps each in quotes, and ORs them into an FTS5 `MATCH`.
  It ranks `-bm25(fact_fts)` descending (scores are higher-is-better), filters by
  `bank_id` and `FactScope`, and applies `top_k`; ties break by fact id. Empty
  queries, zero `top_k`, and empty `Types` scopes return empty without a query.
  Trigram limitations are explicit: substring semantics, no stemming/stopwords,
  two-character terms (e.g. `AI`, `北京`) are ignored, and there is no LIKE
  fallback. Word-level BM25 or a dual index remains a future backend option
  (P5), not part of this task.
- **Acceptance**: proper nouns/numbers match; CJK strings of at least three
  characters match; special characters cannot break the query; short/empty
  queries return empty; scope/bank/top-k are respected; scores are positive.
- **Tests**: term frequency ordering; CJK; punctuation/injection-style inputs;
  scope + bank isolation; top-k; short-term/empty behavior; empty Types scope.

## P2-3 Graph Channel (Spreading Activation)

**Status**: ✅ (2026-11-17)

- **Design**:
```rust
pub struct GraphConfig {
  pub delta: f32,
  pub mu_temporal: f32, pub mu_semantic: f32, pub mu_entity: f32, pub mu_causal: f32,
  pub max_depth: u8, pub max_visits: usize,
}  // defaults: delta 0.5; mu 0.5 / 0.8 / 1.2 / 1.5; depth 2; visits 1000
pub async fn spread(store:&dyn Store, bank:BankId, seeds:&[(FactId,f32)],
                    cfg:&GraphConfig, scope:&FactScope) -> Result<Vec<(FactId,f32)>>;
```
  Activation follows `A(fj)=max(A(fi) * w * delta * mu(kind))`: seeds are the
  initial activations (non-positive/non-finite seeds ignored; duplicates keep
  the max), the output includes the seeds, and results are sorted by activation
  desc then fact id. `max_depth` counts edges from a seed; `max_visits` caps
  expanded nodes; a candidate only re-enqueues when it strictly improves the
  node's activation, so cycles terminate (or are bounded by the caps). `delta
  ∈ (0,1]`; µ values are finite non-negative. `spread` takes the caller's
  `FactScope` and passes it to `neighbors`, so activation cannot leave the target
  network (paper eq. 12 keeps `V` inside the scope).
  `Store::neighbors` is now implemented: outgoing `fact_link` rows grouped by
  kind (max weight) plus derived entity adjacency (weight 1.0), filtered by bank
  and scope, in deterministic order; a fact reachable by two kinds appears once
  per kind so the matching µ applies.
- **Acceptance**: depth/visit caps enforced; deterministic (same input, same
  order); cycles terminate; derived entity adjacency participates; per-kind µ
  is applied.
- **Tests**: chained decay; cycle termination; depth cap; visit cap; entity-only
  adjacency; seed dedup/filtering; neighbor link/entity merge, scope, and
  deterministic ordering; a scoped chain cannot pass through an out-of-scope
  fact; the recall pipeline's graph channel returns no out-of-scope facts.

## P2-4 Temporal Channel

**Status**: ✅ (2026-11-19)

- **Design**:
```rust
pub struct TemporalConfig { pub top_k: usize, pub min_score: f32 }  // defaults: 50 / 0.0
pub async fn temporal_channel(store:&dyn Store, bank:BankId, query:&str,
                              now:Timestamp, parser:&dyn TemporalParser,
                              cfg:&TemporalConfig, scope:&FactScope)
  -> Result<Vec<(FactId,f32)>>;
```
  `TemporalParser::parse` maps the query to a closed range; `None` (no recognizable
  time expression) returns empty without touching the store. The channel calls
  `facts_in_range(bank, range, scope)`, loads those facts, and scores each by
  `1 - |fact_mid - query_mid| / (Δτ/2)`, clamped to `[0, 1]`; a zero-width query
  range floors the denominator at 1 ms so scores stay finite. Facts with an
  unknown occurrence cannot be scored and are dropped; an open interval uses its
  start as the midpoint. Results are sorted by score desc then id, filtered by
  `min_score` (inclusive), and truncated to `top_k`. `top_k = 0` disables the
  channel; a reversed parser range is `Config`.
- **Acceptance**: no time constraint means no activation; closed-interval
  intersection semantics come from `facts_in_range`; UTC-only; scope/bank
  isolation; deterministic ordering.
- **Tests**: proximity ordering and scores; `min_score` + top-k; rule parser
  ("yesterday") end-to-end; no-time query; open intervals and scope; zero-width
  range; disabled channel; config validation.

## P2-5 RRF

**Status**: ✅ (2026-11-19)

- **Design**:
```rust
pub const DEFAULT_RRF_K: u32 = 60;
pub fn rrf(lists: &[Vec<FactId>], k: u32) -> Vec<(FactId, f32)>;
```
  Rank-based fusion (paper eq. 15): each id scores `Σ 1/(k + rank_i)` over the
  lists it appears in, with 1-based positional ranks. A list that does not
  contain the id contributes nothing (no penalty), so raw scores from different
  channels never need calibration. Ranks are summed in `f32`; `k` and rank are
  converted through `u64` so `u32::MAX` cannot overflow. Output contains the
  union of all lists, sorted by fused score descending then id ascending. An id
  repeated inside one list contributes once (the repeat is skipped; later
  entries keep their positional rank).
- **Acceptance**: a single list preserves its order; a missing list contributes
  0; multi-list agreement outranks a single strong hit when the sums say so;
  ordering is deterministic.
- **Tests**: single-list order and scores; crossing lists; missing/empty list;
  equal-score id tie-break; duplicate id inside a list; empty input and a huge
  `k` stay finite.

## P2-6 Rerank Integration

**Status**: ✅ (2026-11-19)

- **Design**:
```rust
pub async fn rerank_top(reranker:&dyn Reranker, query:&str,
                        facts:&[Fact], top_n:usize) -> Vec<Fact>;
```
  Scores only the first `top_n` facts jointly with the query and reorders that
  prefix by score descending, ties broken by the incoming position. The tail
  keeps the fused order; the output has the same length and facts. `top_n = 0`
  disables reranking; `top_n` larger than the list reranks everything. A
  reranker error or a score-count mismatch keeps the original order and logs a
  warning, so recall never fails because the cross-encoder is unavailable.
- **Acceptance**: only the top-N prefix is reordered; the rest is unchanged;
  failures degrade to the fused order; the identity reranker is a no-op.
- **Tests**: identity; reversing fake; tail untouched; error and wrong-length
  fallbacks; zero `top_n`, single/empty lists, oversized `top_n`; equal scores
  keep the incoming order.

## P2-7 Budget Packing

**Status**: ✅ (2026-11-20)

- **Design**:
```rust
pub trait TokenEstimator { fn estimate(&self, text:&str)->usize }   // P0-11
pub struct CharHeuristic;                                         // ceil(chars * ratio)
pub fn pack(facts:&[(Fact,f32)], budget:usize, est:&dyn TokenEstimator)
  -> (Vec<Fact>, usize);
```
  Greedy in final ranked order: add facts while the cumulative estimate stays
  within `budget`, stop at the first overflow, and return the kept facts plus
  tokens used. Packing deliberately does not backfill with smaller lower-ranked
  facts — order is the ranking signal, and skipping ahead would silently promote
  worse evidence. A fact whose text alone exceeds the budget is not truncated
  (truncating generated text would fabricate a partial quote); packing returns
  empty and logs a warning when that happens on the first fact. `budget = 0`
  keeps nothing. Arithmetic uses `checked_add`, so a pathological estimate cannot
  wrap the counter.
- **Acceptance**: cumulative ≤ budget; stops on overflow; first-fact overflow
  returns empty (fixed behavior); order preserved.
- **Tests**: exact fit, one short, one over; no backfilling; `budget=0`; empty
  input; empty text (zero cost); default `CharHeuristic` ratio; overflow guard.

## P2-8 Recall Products and access_count

**Status**: ✅ (2026-11-21)

- **Design**:
```rust
pub enum Channel { Semantic, Bm25, Graph, Temporal }   // snake_case wire names
pub struct RecalledFact { pub fact: Fact, pub score: f32, pub channels: Vec<(Channel, u32)> }
pub struct RecallResult { pub facts: Vec<RecalledFact>, pub tokens_used: usize }
impl RecallResult { pub fn ids(&self)->Vec<FactId>;
                    pub fn channels_of(&self, id:FactId)->Vec<(Channel,u32)>;
                    pub fn is_multi_channel(&self, id:FactId)->bool; }
pub fn sort_channels(channels:&mut Vec<(Channel,u32)>);   // canonical order + dedup
```
  Channel evidence is ordered semantic → bm25 → graph → temporal and deduplicated,
  so the product is deterministic and reflect can inspect multi-channel support
  without re-running retrieval. `increment_access` is implemented as one batched
  `UPDATE … WHERE id IN (...)` per 500 unique ids (deduplicated first): counters
  accumulate across calls, duplicates inside a call count once, unknown ids are
  ignored, and `updated_at` is untouched. Recall calls it best-effort after
  packing (P2-9).
- **Acceptance**: channel overlap is inspectable; `increment_access` is batched
  and idempotent per call; products serialize.
- **Tests**: channel wire names + ordering/dedup; result accessors and overlap;
  JSON round trip; access counter increments/duplicates/unknown ids/empty batch/
  `updated_at` untouched; 1001-id chunked batch.

## P2-9 Recall API

**Status**: ✅ (2026-11-21)

- **Design**:
```rust
pub struct ChannelMask { pub semantic/bm25/graph/temporal: bool }   // ALL, NONE, only(channel)
pub struct RecallOptions {
  pub token_budget: usize, pub channels: ChannelMask, pub per_channel_k: usize,
  pub rerank_top_n: usize, pub rrf_k: u32, pub scope: FactScope,
  pub time_range: Option<TimeRange>,
  pub semantic: SemanticConfig, pub bm25: Bm25Config,
  pub graph: GraphConfig, pub temporal: TemporalConfig,
}  // defaults: 2000 / ALL / 50 / 20 / 60 / All / None
pub async fn recall(store, embedder, reranker, temporal_parser, estimator,
                    bank, query, now, options:&RecallOptions) -> Result<RecallResult>;
impl MemoryBank {
  pub fn with_recall_models(&self, reranker:Arc<dyn Reranker>, parser:Arc<dyn TemporalParser>) -> Self;
  pub fn with_token_estimator(&self, estimator:Arc<dyn TokenEstimator>) -> Self;
  pub async fn recall(&self, bank:BankId, query:&str, options:RecallOptions) -> Result<RecallResult>;
}
```
  `engine::recall::recall` is pure orchestration: embed the query **once**, run the
  enabled channels (each truncated to `per_channel_k`), fuse with RRF, reload the
  fused ids in order, optionally rerank the head, then pack into `token_budget`.
  Defaults bind an `IdentityReranker` and `RuleTemporalParser`; `MemoryBank::recall`
  also performs a best-effort `increment_access` on the returned facts and never
  fails recall when the counter write fails. The graph channel is seeded from the
  semantic list (skipped with a warning when the semantic channel is disabled or
  empty); the temporal channel uses `options.time_range` when present, otherwise
  the injected parser (inactive without a time expression). `ChannelMask::NONE`
  returns an empty result without touching the store.
- **Acceptance**: each channel can be disabled independently; the query is
  embedded only once; RRF → rerank → pack order fixed; budget respected; scope
  and explicit time range honored; access counters are best-effort.
- **Tests**: multi-channel provenance; budget truncation; disabled channels and
  `NONE`; scope isolation; explicit time range; empty bank; missing embedder is
  `Config`; channel-mask/seed/provenance helpers.

## P2-10 Performance Baseline

**Status**: ✅ (2026-11-21)

- **Design**: `crates/store/benches/vector_search.rs` (criterion, `harness = false`).
  Cases: 1k and 10k embedded facts, each with `FactScope::All` (one KNN) and
  `FactScope::Types([World, Opinion])` (one KNN per type, the path P2-1
  exercises). Each case builds an isolated in-memory bank. Run with
  `cargo bench -p store --bench vector_search`; CI skips benches.
- **Recorded numbers** (Apple Silicon, release profile, `--sample-size 10`,
  exact `vec0` scans, k=10, 1536 dims; wall time per `search_vector`):
  | facts | scope | median |
  |---|---|---|
  | 1k | All | 2.26 ms |
  | 1k | World + Opinion | 3.09 ms |
  | 10k | All | 20.97 ms |
  | 10k | World + Opinion | 28.37 ms |
  Latency is linear in the scanned rows (≈2.1 µs per fact), and the multi-type
  path costs roughly one scan per listed type. This is the expected cost of
  exact search; the threshold for moving to an ANN index is P5-10.
- **Acceptance**: numbers recorded for reference (no hard threshold), bench
  runnable.
- **Tests**: `cargo check -p store --benches` (compiles); benches are not part of
  `cargo test`.

---

# P3 · reflect

## P3-1 φ(Θ) Verbalization

**Status**: ❌

- **Design**: `pub fn verbalize(p:&BankProfile)->String`; 5 levels each for S/L/E + β modulation in three bands (0/0.5/1 rules).
- **Acceptance**: stable output for the same profile; β=0 softens preference wording.
- **Tests**: snapshot for extreme combinations.

## P3-2 System Message Assembly

**Status**: ❌

- **Design**: `pub fn system_message(bank:&Bank, evidence:&[RecalledFact])->String` (name/background/φ/numbered evidence block).
- **Acceptance**: evidence is referenceable by number; with no evidence, explicitly states "no relevant memories".
- **Tests**: snapshot; oversized-evidence truncation policy.

## P3-3 Answer + Opinion Extraction

**Status**: ❌

- **Design**: `pub async fn reflect_llm(llm:&dyn Llm, sys:String, user:String)->Result<ReflectResponse>` (schema `reflect`).
- **Acceptance**: opinion confidence ∈[0,1]; empty opinions are valid.
- **Tests**: FakeLlm returns two opinions.

## P3-4 Opinion Persistence

**Status**: ❌

- **Design**: `NewOpinion { id, bank_id, text, confidence, entities: Vec<(EntityId, Option<String>)>, embedding, created_at }`; entity binding via `bind_known_entities(store, bank, text)` (exact/substring matching against canonical names/variants, no LLM calls).
- **Acceptance**: chunk_id=None; confidence required; embedding written.
- **Tests**: 0/2 entities bound; text containing multiple known entities.

## P3-5 Reinforcement Reuse

**Status**: ❌

- **Design**: after reflect, call the same `reinforce()` entry point for new and existing opinions.
- **Acceptance**: no logic fork (one implementation).
- **Tests**: reuse the P1-15 cases.

## P3-6 Evidence Passthrough

**Status**: ❌

- **Design**: `ReflectResult { answer, opinions_formed: Vec<FactId>, recalled: RecallResult }`.
- **Acceptance**: the answer↔evidence id relationship is externally inspectable (no text parsing).
- **Tests**: return-structure assertions.

## P3-7 Behavior Tests

**Status**: ❌

- **Design**: FakeLlm records requests; assert system-prompt differences between high/low-skepticism profiles and β modulation.
- **Acceptance**: prompt assertions stable (snapshots).
- **Tests**: compare two profiles.

## P3-8 reflect API

**Status**: ❌

- **Design**: `impl MemoryBank { pub async fn reflect(&self, bank, query, opts:ReflectOptions)->Result<ReflectResult> }` (internally recall + LLM + persistence).
- **Acceptance**: end-to-end runs; failures retryable (pre-generated opinion ids + single-transaction insert prevent duplicates).
- **Tests**: integration (fake).

---

# P4 · observe and polish

## P4-1 Observation Synthesis Call

**Status**: ❌

- **Design**: `pub async fn synthesize_observations(llm, entity, facts, cfg) -> Result<Vec<ObservationDraft>>` (A.3, 3–7 items, objective third person).
- **Acceptance**: count range enforced; no call for an empty fact set.
- **Tests**: FakeLlm count/content validation.

## P4-2 Evidence-Change Work List

**Status**: ❌

- **Design**: `pending_work.entities_stale_observation`: `entity.last_seen > max(observation.created_at bound to entity)` (SQL aggregation + index).
- **Acceptance**: new evidence triggers; no change means no repeat trigger; capped by parameter.
- **Tests**: new fact → entity enters the list; after synthesis it leaves.

## P4-3 Append-Only Insert

**Status**: ❌

- **Design**: `NewObservation { fact_id, bank_id, text, embedding, entity_ids, created_at }`; one transaction inserts rows + full bindings + FTS.
- **Acceptance**: multi-entity bindings; no chunk_id; old batches not deleted.
- **Tests**: repeated synthesis creates new rows (not replacements).

## P4-4 Observation Retrieval

**Status**: ❌

- **Design**: `observations_of` ordered by `created_at DESC LIMIT n`; facade exposes `entity_profile(bank, entity, limit)`.
- **Acceptance**: recency preferred; coexists with natural recall retrieval.
- **Tests**: ordering assertions across two batches.

## P4-5 GC

**Status**: ❌

- **Design**: `gc_observations(bank, max_age, per_entity_cap)`: delete by age first, then keep the newest `cap` rows per entity binding; same transaction.
- **Acceptance**: non-observation rows unaffected; idempotent.
- **Tests**: age-based and count-based pruning; shared entity rows counted by multiple entities.

## P4-6 sweeper

**Status**: ❌

- **Design**: `spawn_sweeper(cfg)`: periodically run rescan + observation synthesis queue + embedding backfill + GC; dedupe (skip an entity already in flight); jitter.
- **Acceptance**: after a crash and restart it converges automatically; no duplicate synthesis of the same batch.
- **Tests**: after two sweeps, work is empty.

## P4-7 Quality Polish

**Status**: ❌

- **Design**: resolution thresholds/co-occurrence weights, link pruning, and chunking parameters all configurable; add data-driven tuning scripts (examples).
- **Acceptance**: configuration changes require no code changes.
- **Tests**: config snapshots/defaults.

## P4-8 P4 Integration Tests

**Status**: ❌

- **Tests**: observation append/reclaim, GC × recall interaction, entity profile API.

## P4-9 Entity Merge

**Status**: ✅ (2026-11-16)

- **Content**: a manual/programmatic consolidation path for duplicate canonical
  entities created by wording differences (the resolution threshold prevents
  false merges, at the cost of possible duplicates).
- **Design**:
```rust
async fn merge_entities(&self, bank:BankId, from:EntityId, into:EntityId,
                        now:Timestamp) -> Result<()>;
impl MemoryBank { pub async fn merge_entities(&self, bank, from, into) -> Result<()> }  // injected Clock
```
  One maintenance transaction: delete source bindings whose `(fact_id, into)`
  already exist; move the remaining `fact_entity` rows to `into`; merge
  `entity_type` (specific wins over `Other`); recompute
  `mention_count`/`first_seen`/`last_seen` from the surviving world/experience
  bindings; append `from` (and its own `merged_from` history) to `into.attrs.merged_from`;
  delete the `from` row.
  `from == into` is `Config`; missing/cross-bank ids are `NotFound`. Observation
  and opinion bindings move with the rest; `fact_link` is untouched because
  entity adjacency is derived.
- **Acceptance**: no dangling `fact_entity` rows; surviving stats equal the
  world/experience binding count; source row and its canonical name are gone;
  provenance is inspectable in `attrs`.
- **Tests**: binding move + same-fact dedup; stats recompute; type upgrade from
  `Other`; observation binding move; provenance chain; self-merge and cross-bank
  rejection; facade clock stamping.

---

# P5 · adapters and evaluation

## P5-1 PG Migration + Adapter

**Status**: ❌ (deferred — SQLite only for now; revisit when a second backend is needed)

- **Design**: a separate `postgres` module (sqlx::Postgres), jsonb/timestamptz mapping; `vector` type as an optional feature; dialect differences isolated in the module.
- **Acceptance**: two implementations of the same trait; contract tests pass.
- **Tests**: testcontainers or a local PG (optional in CI).

## P5-2 Dual-Backend Store Contract Tests

**Status**: ❌ (deferred — depends on P5-1)

- **Design**: `store::testing::contract(factory: impl BackendFactory)`, covering method semantics (claim atomicity, conflict skip, adjacency merging, statistics maintenance).
- **Acceptance**: both backends green.
- **Tests**: the contract suite itself.

## P5-3 Real Reranker

**Status**: ❌

- **Design**: `llm::HttpReranker { endpoint, model, timeout }`; batch document scoring; fall back to original order on failure.
- **Acceptance**: interchangeable with the stub.
- **Tests**: mock HTTP (wiremock).

## P5-4 Real TemporalParser

**Status**: ❌

- **Design**: rule library (multilingual dates/relative expressions) + optional t5-small fallback (HTTP or ONNX).
- **Acceptance**: parse rate on common expressions; unparseable returns None.
- **Tests**: table-driven date corpus.

## P5-5 Concurrency and Configuration Cleanup

**Status**: ❌

- **Design**: worker concurrency, batch sizes, timeouts/retries, and log sampling all configurable; tracing spans (bank/chunk/fact).
- **Acceptance**: no lock-contention hotspots under load; retry metrics visible.
- **Tests**: config-combination smoke tests.

## P5-6 LongMemEval Harness

**Status**: ❌

- **Design**: load data → retain per session → per-question recall+reflect → LLM-as-judge (paper A.4 prompts) → metric aggregation.
- **Acceptance**: reproducible run (fake mode validates the pipeline first).
- **Tests**: small-sample golden.

## P5-7 LoCoMo Harness

**Status**: ❌

- **Design**: multi-session dialogues + image-field ignoring policy → same as P5-6.
- **Acceptance**: unified metric output format.
- **Tests**: small sample.

## P5-8 Documentation and Examples

**Status**: ❌

- **Design**: README quick start; examples: `retain_basic`, `recall_query`, `reflect_with_profile`, `custom_store` (demonstrating port replacement).
- **Acceptance**: `cargo run --example` works (fake mode, no API key).
- **Tests**: examples compile + smoke run.

## P5-9 Local Model Adapter

**Status**: ❌

- **Content**: first-class local-model support: OpenAI-compatible local endpoints plus an optional in-process adapter.
- **Design**:
  - HTTP path: the existing adapter works unchanged against Ollama / vLLM / llama.cpp server / SGLang / LM Studio (`base_url` + `model`, `api_key` optional, timeout configured per adapter).
  - Optional `llm/local` feature: in-process `Llm` / `Embedder` adapters (candle, ONNX Runtime, or mistral.rs); blocking inference runs on `spawn_blocking` or a dedicated bounded worker pool, never on the async runtime thread; heavy dependencies stay behind the feature flag.
  - Capability mapping: when `supports_json_schema()` is `false`, render the schema into the prompt and extract JSON from the completion; use grammar/guided decoding (GBNF, guided JSON) where the runtime supports it.
  - Bounded repair: on a validation failure, feed the error plus schema back to the model up to N times; transient provider failures → `Retryable`, exhausted repair → `Fatal`.
  - Timeout, concurrency, batch size, and embedding normalization are adapter configuration, not schema or engine concerns.
- **Acceptance**: a local endpoint (or in-process model) satisfies the same `Llm` / `Embedder` contract suite as the HTTP provider; no blocking calls on the async runtime; no schema change; default build does not pull in model-runtime dependencies.
- **Tests**: wiremock for the HTTP path; in-process adapter `#[ignore]`d when weights are unavailable; repair-path unit tests (invalid JSON → repaired → valid; repair exhausted → `Fatal`).

## P5-10 Vector Index Acceleration

**Status**: ❌

- **Content**: when exact sqlite-vec search is no longer enough, add an approximate index (HNSW) and benchmark it.
- **Current implementation (P1-13)**: migration `0001` creates `fact_vec` (`fact_id`, `bank_id`, `embedding float[1536] distance_metric=cosine`) as the sole vector storage; a fact without a row is pending backfill, and a delete trigger removes the vector with its fact. Queries are exact, run in SQLite's C path, and no longer scan/decode BLOBs in Rust. Changing to a different embedding dimension requires rebuilding the table.
- **Design**: the engine only sees `Store::search_vector`, so acceleration is adapter-internal:
  - HNSW via `usearch` (in-process) or an external vector store: sub-linear queries at the cost of approximate recall; index persistence, tombstones/deletes, and rebuilds on embedding-model cutover are adapter responsibilities.
  - PostgreSQL + pgvector HNSW becomes available with the deferred PG adapter (P5-1).
  - Index maintenance is automatic: the schema triggers mirror any `fact` insert/update/delete into `fact_vec`, so tx2 needs no special handling.
- **Acceptance**: the same `Store` contract tests pass; the exact sqlite-vec path remains available as a fallback; recall/latency measured on a representative bank before switching; model cutover rebuilds the index.
- **Tests**: adapter contract suite; benchmark comparing exact (vec0) vs HNSW recall/latency; index rebuild on generation cutover.

---

# Delivery Notes

- One commit per step; commit message `P<phase>-<index>: <task>`;
- P0/P1 are the mainline and should be **reviewed per task before merging**; from P2 onward, retrieval and reflection can proceed in parallel;
- End of each phase: `cargo test --workspace` + phase integration tests + clippy/fmt;
- Settle open semantics before starting the relevant task: §7.A#2 (confidence), causal normalization/relation narrowing, token-estimator implementation.
