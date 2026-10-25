# memorybank Implementation Steps (Phased Task List)

Based on `design.md` (implementation design) and `memory-schema.md` (finalized logical schema).

> **Status convention**: each task's `**Status**` line defaults to ❌; when complete, change it to ✅ (a date or commit hash is recommended).
> Current progress: **P0 7/12 · P1 0/19 · P2 0/10 · P3 0/8 · P4 0/8 · P5 0/8 (7/65 total)**

## Common Conventions (apply to every step; not repeated below)

- **Async**: ports and orchestration are all `async` (`async-trait`); retain's "synchronous semantics" means `.await` until this call's chunks are all done — it does not block a thread.
- **Errors**: `engine::Error` (Retryable/Fatal/Config/NotFound/Conflict); workers retry only Retryable; tests may use `anyhow` helpers.
- **IDs**: UUIDv7, allocated in **engine before the transaction** (`FactId::new()`); `Store` only persists them — avoiding the backtracking dependency of "insert before you get an id".
- **Testing**: unit tests beside the module; integration tests in `crates/*/tests/`; all time goes through an injected `Clock`; an isolated in-memory SQLite database per test.
- **Configuration**: `engine::Config` (chunking/retrieval/linking/reinforcement/scheduling thresholds), serde-deserialized; engine reads no environment variables (the facade injects them).

---

# P0 · Foundation

## P0-1 Workspace Skeleton

**Status**: ✅ (2026-10-25)

- **Content**: root `Cargo.toml` (virtual workspace), `crates/{memorybank,engine,store,llm}`, `rustfmt.toml`, clippy lints, CI script.
- **Design**: workspace members; features: `store/sqlite` (default), `store/postgres` (later); `llm/http`.
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

**Status**: ✅ (2026-10-25)

- **Content**: all core enums/row structs.
- **Design**:
```rust
pub enum FactType { World, Experience, Opinion, Observation }
pub enum LinkType { Temporal, Semantic, Causal }
pub enum ChunkStatus { Pending, Extracting, Done, Failed }
pub enum CausalRelation { Causes, CausedBy, Enables, Prevents }   // normalization pending decision
pub enum EntityKind { Person, Organization, Location, Product, Concept, Other }
pub enum SourceStatus { Received, Chunked }

pub struct BankProfile { pub background: String, pub skepticism: u8, pub literalism: u8, pub empathy: u8, pub bias_strength: f32 }
impl BankProfile { pub fn validate(&self)->Result<()>; }   // 1..=5, 0..=1

pub struct Bank { id, name: String, profile: BankProfile, created_at, updated_at }
pub struct Source { id, bank_id, title: Option<String>, payload_text: String, status, created_at, updated_at }
pub struct Chunk { id, bank_id, source_id, chunk_index: u32, text: String, content_hash: String,
                   status: ChunkStatus, time_start: Option<Timestamp>, time_end: Option<Timestamp>, created_at, updated_at }
pub struct Fact { id, bank_id, fact_type: FactType, text: String, chunk_id: Option<ChunkId>,
                  embedding: Option<Vec<f32>>, occurred_start/occurred_end/mentioned_at: Option<Timestamp>,
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

**Status**: ✅ (2026-10-25)

- **Content**: LLM round-trip contracts + JSON Schema (schemars).
- **Design**:
```rust
pub struct ExtractedFact { pub what: String, pub when: String, pub where_: String, pub who: String,
  pub why: String, pub fact_type: FactType, pub occurred_start: Option<Timestamp>,
  pub occurred_end: Option<Timestamp>, pub mentioned_at: Option<Timestamp>,
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
  pub occurred_start/occurred_end/mentioned_at: Option<Timestamp>,
  pub confidence: Option<f32>, pub attrs: Option<serde_json::Value>, pub created_at: Timestamp }
pub struct NewBinding { pub bank_id: BankId, pub fact_id: FactId, pub entity_id: EntityId,
  pub mention_text: Option<String>, pub created_at: Timestamp }
pub struct NewLink { pub id: LinkId, pub bank_id: BankId, pub from: FactId, pub to: FactId,
  pub link_type: LinkType, pub relation: Option<CausalRelation>, pub weight: f32,
  pub created_at: Timestamp }
pub struct EntityStatUpdate { pub entity_id: EntityId, pub add: bool, pub at: Option<Timestamp>,
  pub updated_at: Timestamp }
pub struct OpinionConfidenceUpdate { pub opinion_id: FactId, pub new_confidence: f32,
  pub new_text: Option<String>, pub updated_at: Timestamp }
pub struct ChunkCommit { pub chunk_id: ChunkId, pub completed_at: Timestamp,
  pub facts: Vec<NewFact>, pub entities: Vec<Entity>, pub bindings: Vec<NewBinding>,
  pub links: Vec<NewLink>, pub entity_stats: Vec<EntityStatUpdate>,
  pub opinion_updates: Vec<OpinionConfidenceUpdate> }
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
  pub last_mention_at: Option<Timestamp> }

#[async_trait] pub trait Store: Send+Sync {
  // bank / inbox
  async fn create_bank(&self, bank:Bank)->Result<()>;
  async fn load_bank(&self, id:BankId)->Result<Bank>;
  async fn create_source(&self, source:Source)->Result<()>;
  async fn append_chunks(&self, src:SourceId, chunks:Vec<NewChunk>)->Result<usize>;  // inserted count
  async fn claim_chunk(&self, bank:BankId, now:Timestamp, stale_after:Duration)->Result<Option<ClaimedChunk>>;
  async fn release_chunk(&self, chunk:ChunkId)->Result<()>;
  async fn commit_chunk(&self, c:ChunkCommit)->Result<()>;
  // retrieval
  async fn search_vector(&self, bank:BankId, q:&[f32], k:usize)->Result<Vec<(FactId,f32)>>;
  async fn search_bm25(&self, bank:BankId, q:&str, k:usize)->Result<Vec<(FactId,f32)>>;
  async fn neighbors(&self, bank:BankId, from:FactId)->Result<Vec<Neighbor>>;
  async fn facts_in_range(&self, bank:BankId, r:TimeRange)->Result<Vec<FactId>>;
  async fn fetch_facts(&self, ids:&[FactId])->Result<Vec<Fact>>;
  async fn increment_access(&self, ids:&[FactId])->Result<()>;
  // entity / opinion / observation / maintenance
  async fn ensure_entity(&self, new:NewEntity)->Result<(EntityId,bool)>;   // (id, created)
  async fn entity_candidates(&self, bank:BankId, mention:&str)->Result<Vec<EntityCandidate>>;
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
  - `commit_chunk` is the only transaction entry for extraction products: facts / bindings / links / entity stats / opinion updates / status→done (`completed_at`) commit together. If a listed entity hits `UNIQUE(bank_id, canonical_name)`, the store reuses the existing row and rewrites the bindings that referenced the new id.
  - `append_chunks` skips any chunk whose `(bank_id, content_hash)` already exists and returns the inserted count.
  - `neighbors` merges `fact_link` rows with adjacency derived from `fact_entity` (weight 1.0, `NeighborKind::Entity`); the pairwise closure is never materialized. Four kinds are reported because P2-3 applies a distinct `mu` per kind.
  - `insert_opinion` writes the opinion row plus its `fact_entity` bindings in one transaction; `insert_observations` appends observation rows plus bindings, never replacing an older batch.
  - `pending_work` derives all lists in-database; `update_embeddings` is the backfill write path (the engine owns the `Embedder`, so the store cannot re-embed by itself); `increment_access` batches counter bumps and leaves `updated_at` untouched.
  - `gc_observations` deletes observation rows older than `now - max_age`, then keeps the newest `per_entity_cap` rows per entity binding, in one transaction.
- **Acceptance**: the trait is object-safe; DTOs leak no backend types.
- **Tests**: compile-time object-safety test; DTO serde.

## P0-8 Store Shape Finalization

**Status**: ❌

- **Content**: land the single-trait decision + a note on the splitting contingency (read/write/maintenance).
- **Acceptance**: `let s: Box<dyn Store>` compiles; the doc records the trigger for splitting (e.g. a read-only backend).
- **Tests**: dyn injection smoke test.

## P0-9 Migration 0001 (SQLite)

**Status**: ❌

- **Content**: 7-table DDL + FTS + indexes + constraints.
- **Design**:
  - Tables follow the schema fields; FTS5: `CREATE VIRTUAL TABLE fact_fts USING fts5(fact_id UNINDEXED, text, tokenize='trigram')` (maintained manually);
  - Unique: `chunk(bank_id, content_hash)`, `entity(bank_id, canonical_name)`, `fact_link(from_fact_id, to_fact_id, link_type, relation)`;
  - CHECK: enum domains; `(fact_type='opinion' AND chunk_id IS NULL) => confidence IS NOT NULL`;
  - Indexes: `fact(bank_id)`, `fact(chunk_id)`, `fact(bank_id, occurred_start)`, `source(bank_id,status)`, `fact_entity(bank_id, entity_id)`, `fact_entity(bank_id, fact_id)`, `fact_link(bank_id, from_fact_id)`, `fact_link(bank_id, to_fact_id)`.
- **Acceptance**: a line-by-line checklist against `memory-schema.md` §3 in the migration header.
- **Tests**: constraint violations (duplicate hash, duplicate canonical name, missing confidence); FTS insert/query.

## P0-10 Migration Runner + Test DB Helper

**Status**: ❌

- **Content**: `store::migrate(&SqlitePool)`; `store::testing::memory_pool()`.
- **Acceptance**: migrations are idempotent (run twice); each test database is isolated.
- **Tests**: run migrate twice; basic concurrent connections.

## P0-11 Fakes

**Status**: ❌

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

**Status**: ❌

- **Content**: bank→source→chunks→claim/release integration test.
- **Acceptance**: all green; `cargo doc` has no broken links.
- **Tests**: `crates/store/tests/smoke.rs`.

---

# P1 · retain

## P1-1 Payload Normalization

**Status**: ❌

- **Content**: turn caller input into canonical text + a turn view.
- **Design**:
```rust
pub struct RawTurn { pub speaker: String, pub text: String, pub at: Option<Timestamp>, pub session: Option<u32> }
pub enum RawPayload { Transcript { title: Option<String>, turns: Vec<RawTurn> }, Document { title: Option<String>, text: String } }
pub struct TurnSpan { pub byte_start: usize, pub byte_end: usize, pub at: Option<Timestamp>, pub session: Option<u32> }
pub struct NormalizedPayload { pub text: String, pub spans: Vec<TurnSpan> }
pub fn normalize(p: RawPayload) -> NormalizedPayload   // "[<ts>] <speaker>: <text>\n"
```
- **Acceptance**: Document passes through untouched; Transcript preserves time/session markers; spans are invertible (slice == turn text).
- **Tests**: empty turns, CJK, extremely long single turn.

## P1-2 Chunker

**Status**: ❌

- **Content**: session alignment + window fallback + hashing.
- **Design**:
```rust
pub struct ChunkConfig { pub session_gap: Duration, pub max_tokens: usize, pub min_tokens: usize, pub estimator: TokenEstimatorKind }
pub struct RawChunk { pub index: u32, pub text: String, pub content_hash: String,
                      pub time_start: Option<Timestamp>, pub time_end: Option<Timestamp> }
pub fn chunk(p: &NormalizedPayload, cfg: &ChunkConfig, est: &dyn TokenEstimator) -> Vec<RawChunk>
pub fn chunk_hash(text: &str) -> String   // whitespace-normalized + SHA-256
```
- **Acceptance**: deterministic (same input, same output); never emits empty chunks; oversized sessions are window-split without crossing "obvious session boundaries".
- **Tests**: multiple sessions, one oversized turn, whitespace-only differences hash equal.

## P1-3 `create_source` (tx0)

**Status**: ❌

- **Content**: Store implementation + transaction boundary.
- **Acceptance**: payload_text persisted, status=received; no half-writes before the crash point.
- **Tests**: read-back consistency; duplicated callers each get a new row (no meaningful external id).

## P1-4 `append_chunks` (tx1)

**Status**: ❌

- **Content**: batch insert + conflict skip + source status advance.
- **Design**: `INSERT … ON CONFLICT(bank_id,content_hash) DO NOTHING`; same transaction `UPDATE source SET status='chunked'`.
- **Acceptance**: when every chunk conflicts, 0 rows are inserted but the source still becomes chunked (zero-chunk ledger row).
- **Tests**: same-source double ingest; identical content across sources; partial conflicts.

## P1-5 `claim_chunk`

**Status**: ❌

- **Content**: atomic claim + stale reclaim.
- **Design**:
```sql
UPDATE chunk SET status='extracting', updated_at = ?1
WHERE id = (SELECT id FROM chunk WHERE bank_id=?2 AND status='pending'
            ORDER BY created_at LIMIT 1)
  AND status='pending'
RETURNING *;
-- stale reclaim: UPDATE chunk SET status='pending' WHERE status='extracting' AND updated_at < ?old
```
- **Acceptance**: with two concurrent workers only one claims; stale extracting rows are reset.
- **Tests**: 10 concurrent claims, only 1 succeeds; reset after time rewind.

## P1-6 `release_chunk`

**Status**: ❌

- **Content**: return to pending on failure.
- **Acceptance**: status=pending, updated_at refreshed; repeated failures keep the semantics.
- **Tests**: after release, the chunk can be claimed again.

## P1-7 Extraction Call

**Status**: ❌

- **Content**: A.1 prompt + schema + parse validation.
- **Design**:
```rust
pub async fn extract_facts(llm:&dyn Llm, chunk:&Chunk, cfg:&ExtractionConfig) -> Result<ExtractionReport>
pub struct ExtractionReport { pub facts: Vec<ExtractedFact>, pub discarded: usize }
// prompt: full A.1 text (system), user=chunk.text; schema_name="fact_extraction"
// validation: non-empty what; valid fact_type; causal.target_fact_index < facts.len()
pub fn validate_extracted(f: &ExtractedFact) -> Result<()>
```
- **Acceptance**: invalid single facts are discarded and counted; call-level retry only for Retryable.
- **Tests**: FakeLlm returns 3 facts with 1 bad; empty result; oversized text truncation policy.

## P1-8 Time and Augmented Text

**Status**: ❌

- **Content**: time normalization + human-readable prefix + embedding-input assembly.
- **Design**:
```rust
pub fn normalize_time(extracted:&mut ExtractedFact, chunk:&Chunk, clock:&dyn Clock) -> Result<()>
// missing occurred → use chunk.time_start/end; missing mentioned_at → chunk interval midpoint / creation time
pub fn human_readable(ts: Timestamp) -> String            // "Saturday, June 9, 2024"
pub fn embedding_input(text:&str, ts: Option<Timestamp>) -> String  // "{human}. {text}"
```
- **Acceptance**: never fabricate time (None stays None); stored text == embedding input (the same string).
- **Tests**: relative times are already normalized by the LLM (this step only fills defaults); prefix format.

## P1-9 Entity Resolution ρ

**Status**: ❌

- **Content**: exact hit → three-feature fuzzy match → ensure_entity.
- **Design**:
```rust
pub struct ResolveWeights { pub alpha: f32, pub beta: f32, pub gamma: f32, pub threshold: f32 }
pub struct MentionContext { pub co_mentioned: Vec<EntityId>, pub at: Option<Timestamp> }
pub async fn resolve(store:&dyn Store, bank:BankId, mention:&EntityMention,
                     ctx:&MentionContext, w:&ResolveWeights) -> Result<EntityId>
pub fn normalize_name(s:&str)->String                    // lowercase/trim/whitespace collapse
pub fn string_sim(a:&str,b:&str)->f32                    // normalized Levenshtein similarity
pub fn cooccurrence_sim(cand:&EntityCandidate, ctx:&MentionContext)->f32
pub fn temporal_sim(cand:&EntityCandidate, at: Option<Timestamp>)->f32
```
- **Acceptance**: the same mention always resolves to the same entity; creation conflicts fall back to `ensure_entity`'s UNIQUE.
- **Tests**: exact/variant/new-entity paths; concurrent creation of the same name yields one row; empty mention rejected.

## P1-10 Entity Statistics

**Status**: ❌

- **Content**: apply `EntityStatUpdate` inside the commit.
- **Design**: `mention_count += 1`; `first_seen = min`; `last_seen = max` (conversation clock).
- **Acceptance**: consistent with the binding-row count (tests assert COUNT).
- **Tests**: insert/re-extraction replace (−1/+1); NULL time does not update seen.

## P1-11 Embedding Generation

**Status**: ❌

- **Content**: batch embedding + dimension validation + failure degradation.
- **Design**: `pub async fn embed_facts(emb:&dyn Embedder, texts:&[String], batch:usize) -> Vec<Option<Vec<f32>>>`; validate `len==dim()`; failure becomes None (runtime backfill).
- **Acceptance**: dimension mismatch is a Config error; failure does not block tx2.
- **Tests**: batch splitting; single-batch failure → None rows.

## P1-12 Temporal Link Construction

**Status**: ❌

- **Content**: interval proximity → two rows.
- **Design**:
```rust
pub fn temporal_weight(gap: Duration, sigma: Duration) -> f32   // exp(-gap/sigma)
pub async fn build_temporal_links(store:&dyn Store, bank:BankId, new_facts:&[Fact],
                                  cfg:&LinkConfig) -> Result<Vec<NewLink>>
// candidates: interval distance < window; sort by weight and keep the top-K pairs; write from→to and to→from per pair
```
- **Acceptance**: reciprocal pairs always complete (canonical-key count is 2); pruning leaves no one-sided rows; rows without time create no edges.
- **Tests**: three-fact time window; pruning boundaries; weight formula unit test.

## P1-13 Semantic Link Construction

**Status**: ❌

- **Content**: directed top-k.
- **Design**: `pub async fn build_semantic_links(store, bank, new_facts, cfg) -> Result<Vec<NewLink>>`; for each new fact: `search_vector` (exclude self, cos≥θs), take top-K; each edge `from=new, to=cand, weight=cos`.
- **Acceptance**: directed single rows; no B→A unless B also selects A; within budget.
- **Tests**: deterministic distribution with budget 2; self excluded; θs filtering.

## P1-14 Causal Link Construction

**Status**: ❌

- **Content**: map extraction output.
- **Design**: `pub fn causal_links(extracted:&[ExtractedFact], ids:&[FactId]) -> Vec<NewLink>`; `target_fact_index → ids[i]`; single row. Normalization per the decision (currently: keep as-is + relation).
- **Acceptance**: out-of-range indices dropped; duplicate pairs deduplicated; strength does not enter weight (paper w=1.0).
- **Tests**: one-way causality between two facts; chained causality; out-of-range.

## P1-15 Opinion Reinforcement

**Status**: ❌

- **Content**: candidates → Assess → confidence update.
- **Design**:
```rust
pub struct ReinforcementConfig { pub alpha: f32, pub sim_threshold: f32, pub max_candidates: usize }
pub async fn reinforce(store:&dyn Store, llm:&dyn Llm, bank:BankId,
                       new_facts:&[Fact], cfg:&ReinforcementConfig)
  -> Result<Vec<OpinionConfidenceUpdate>>
// candidates: entity overlap ∪ sim(v_o,v_f)>θ (search_vector restricted to opinions)
// Assess schema: {"verdict":"reinforce|weaken|contradict|neutral","reason":"…"}
// c' = min(c+α,1) / max(c-α,0) / max(c-2α,0) / c
```
- **Acceptance**: four-verdict rules unit-tested; neutral produces no update; contradict may rewrite text; everything is atomic with tx2.
- **Tests**: FakeLlm four verdicts; both candidate paths (entity overlap / vector recall); α boundaries (0/1).

## P1-16 `commit_chunk` (tx2)

**Status**: ❌

- **Content**: commit every product in one transaction.
- **Design**: insert order entity→fact→fact_entity→fact_link→FTS→stat updates→opinion updates→chunk.status=done; any failure rolls back.
- **Acceptance**: no residual rows after failure; after success, facts/edges/statistics are consistent.
- **Tests**: inject failure (e.g. FTS unique conflict) and verify full rollback; replaying the same commit is rejected (status already done).

## P1-17 retain API (Synchronous Semantics)

**Status**: ❌

- **Content**: facade method.
- **Design**:
```rust
pub struct RetainOptions { pub wait_until_done: bool }   // v1 always true
pub struct RetainReport { pub source_id: SourceId, pub chunks_total: usize,
                          pub chunks_skipped: usize, pub facts_written: usize }
impl MemoryBank { pub async fn retain(&self, bank:BankId, payload:RawPayload, opts:RetainOptions) -> Result<RetainReport> }
```
- **Acceptance**: on return, this call's chunks are all done (or skipped); error classes propagated with context.
- **Tests**: end-to-end (fake LLM/embedder) produces the expected fact count.

## P1-18 worker + rescan

**Status**: ❌

- **Content**: driven by derived work lists.
- **Design**:
```rust
pub struct RescanReport { pub sources_chunked: usize, pub chunks_processed: usize, pub failed: usize }
impl MemoryBank { pub async fn rescan(&self, bank:BankId) -> Result<RescanReport>;
                  pub fn spawn_worker(&self, cfg:WorkerConfig) -> JoinHandle<()>; }
```
- **Acceptance**: status=received sources are re-chunked; pending/stale-extracting chunks are re-run; failed is not retried automatically (ops reset).
- **Tests**: simulate interruption (manually set status), then rescan converges.

## P1-19 retain Integration Tests

**Status**: ❌

- **Content**: end-to-end matrix.
- **Acceptance/Tests**:
  - re-ingest idempotency (zero new facts); incremental re-ingest (only new chunks processed);
  - interrupted extraction → rescan recovers; tx2 failure → no partial state;
  - two-row invariant (SQL asserts count=2); entity statistics match bindings.

---

# P2 · recall

## P2-1 Semantic Channel

**Status**: ❌

- **Design**: implement `search_vector`: `SELECT id, embedding FROM fact WHERE bank_id=? AND embedding IS NOT NULL`, cosine in Rust, maintain a top-k min-heap; `k`/`min_cos` from configuration.
- **Acceptance**: matches the brute-force reference implementation; NULL rows skipped.
- **Tests**: ordering assertions on a known vector set; empty bank returns empty.

## P2-2 BM25 Channel

**Status**: ❌

- **Design**: `search_bm25`: split the query on whitespace/punctuation and escape terms (wrap in `"term"` quotes), `SELECT fact_id, bm25(fact_fts) AS s … JOIN fact ON … WHERE bank_id=? ORDER BY s LIMIT k`.
- **Acceptance**: CJK trigram substrings match; empty query returns empty.
- **Tests**: proper nouns/numbers match; injection-style special characters do not error.

## P2-3 Graph Channel (Spreading Activation)

**Status**: ❌

- **Design**:
```rust
pub struct GraphConfig { pub delta: f32, pub mu_temporal/mu_semantic/mu_entity/mu_causal: f32,
                         pub max_depth: u8, pub max_visits: usize }
pub async fn spread(store:&dyn Store, bank:BankId, seeds:&[(FactId,f32)], cfg:&GraphConfig)
  -> Result<Vec<(FactId,f32)>>   // A(fj)=max(A(fi)*w*delta*mu(ℓ))
```
- **Acceptance**: depth/visit caps enforced; deterministic (same input, same order).
- **Tests**: chained-graph activation decay; cycles terminate; derived entity adjacency participates.

## P2-4 Temporal Channel

**Status**: ❌

- **Design**: `TemporalParser.parse` → `facts_in_range` → proximity score `1 - |midDiff|/(range/2)`.
- **Acceptance**: no time constraint means no activation; interval endpoint semantics fixed (closed interval).
- **Tests**: "last week" query; no results; timezone normalization to UTC.

## P2-5 RRF

**Status**: ❌

- **Design**: `pub fn rrf(lists:&[Vec<FactId>], k:u32)->Vec<(FactId,f32)>`; union, stable sort (score desc, id asc).
- **Acceptance**: a single list preserves its order; a missing list contributes 0.
- **Tests**: two crossing lists; tie-breaking stability.

## P2-6 Rerank Integration

**Status**: ❌

- **Design**: `pub async fn rerank_top(reranker:&dyn Reranker, query:&str, facts:&[Fact], top_n:usize)`; on error fall back to the original order and log.
- **Acceptance**: only top-N reordered; the rest unchanged.
- **Tests**: Identity preserves; reversing fake takes effect.

## P2-7 Budget Packing

**Status**: ❌

- **Design**:
```rust
pub trait TokenEstimator { fn estimate(&self, text:&str)->usize }
pub struct CharHeuristic;      // ceil(chars * ratio), configurable
pub fn pack(facts:&[(Fact,f32)], budget:usize, est:&dyn TokenEstimator) -> (Vec<Fact>, usize)
```
- **Acceptance**: cumulative ≤ budget; stops on overflow; if the first fact alone exceeds the budget, return empty and warn (fixed behavior).
- **Tests**: boundary cases budget=0 / exactly equal / one short.

## P2-8 Recall Products and access_count

**Status**: ❌

- **Design**:
```rust
pub struct RecalledFact { pub fact: Fact, pub score: f32, pub channels: Vec<(Channel, u32 /*rank*/)> }
pub struct RecallResult { pub facts: Vec<RecalledFact>, pub tokens_used: usize }
```
- **Acceptance**: channels mark overlaps; `increment_access` is batched.
- **Tests**: multi-channel hit annotation.

## P2-9 Recall API

**Status**: ❌

- **Design**:
```rust
pub struct RecallOptions { pub token_budget: usize, pub channels: ChannelMask,
                           pub time_range: Option<TimeRange>, pub rerank_top_n: usize, pub per_channel_k: usize }
impl MemoryBank { pub async fn recall(&self, bank:BankId, query:&str, opts:RecallOptions) -> Result<RecallResult> }
```
- **Acceptance**: each of the four channels can be disabled independently; the query is embedded only once; total latency instrumented.
- **Tests**: hit assertions for 10 questions on a fixed corpus (fake embedder semantics can be constructed).

## P2-10 Performance Baseline

**Status**: ❌

- **Design**: `benches/` (criterion) or example timing; record brute-force cosine latency at 1k/10k facts.
- **Acceptance**: numbers recorded under `design/` for reference (no hard threshold).
- **Tests**: benchmarks runnable (CI skips them).

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

---

# P5 · adapters and evaluation

## P5-1 PG Migration + Adapter

**Status**: ❌

- **Design**: a separate `postgres` module (sqlx::Postgres), jsonb/timestamptz mapping; `vector` type as an optional feature; dialect differences isolated in the module.
- **Acceptance**: two implementations of the same trait; contract tests pass.
- **Tests**: testcontainers or a local PG (optional in CI).

## P5-2 Dual-Backend Store Contract Tests

**Status**: ❌

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

---

# Delivery Notes

- One commit per step; commit message `P<phase>-<index>: <task>`;
- P0/P1 are the mainline and should be **reviewed per task before merging**; from P2 onward, retrieval and reflection can proceed in parallel;
- End of each phase: `cargo test --workspace` + phase integration tests + clippy/fmt;
- Settle open semantics before starting the relevant task: §7.A#2 (confidence), causal normalization/relation narrowing, whether to split the Store trait, token-estimator implementation.
