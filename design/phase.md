# memorybank 实现步骤（分阶段任务清单）

依据：`design.md`（实现设计）、`memory-schema.md`（逻辑表结构定稿）。

> **状态标记约定**：每个任务下的 `**状态**` 行用 **✅ 已完成 / ❌ 未完成** 标记；完成时把 ❌ 改为 ✅（建议附日期/提交号）。
> 当前进度：**P0 0/12 · P1 0/19 · P2 0/10 · P3 0/8 · P4 0/8 · P5 0/8（合计 0/65，全部 ❌）**

## 通用约定（所有步骤默认遵守，不再重复）

- **异步**：port 与编排全部 `async`（`async-trait`）；`retain` 的"同步语义"指 `.await` 直到本次 chunks 全 done，不是阻塞线程。
- **错误**：`engine::Error`（Retryable/Fatal/Config/NotFound/Conflict）；worker 只重试 Retryable；测试用 `anyhow` 辅助。
- **ID**：UUIDv7，engine 在**事务前**分配（`FactId::new()`），Store 只负责持久化——避免"插入才能拿到 id"的回溯依赖。
- **测试**：单测在模块内；集成测试在 `crates/*/tests/`；所有时间走 `Clock` 注入；SQLite 内存库每测试独立。
- **配置**：`engine::Config`（分块/检索/建链/强化/调度阈值），serde 反序列化，engine 不读环境变量（由 facade 注入）。

---

# P0 · 地基

## P0-1 Workspace 骨架

**状态**：❌

- **内容**：根 `Cargo.toml`（workspace + package `memorybank`）、`crates/{engine,store,llm}`、`rustfmt.toml`、clippy lints、CI 脚本。
- **设计**：workspace members；features：`store/sqlite`（默认）、`store/postgres`（后置）；`llm/http`。
- **验收**：`cargo check/clippy/fmt --workspace` 全绿；`cargo tree` 无反向依赖（engine 不依赖 store/llm）。
- **测试**：CI 跑 `cargo test --workspace`。

## P0-2 ID 与时间基元

**状态**：❌

- **内容**：5 个 ID newtype + 时间别名。
- **设计**：
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
- **验收**：serde 往返；UUIDv7 按创建时间可排序。
- **测试**：roundtrip；连续 new() 的字典序单调。

## P0-3 领域枚举与结构

**状态**：❌

- **内容**：全部核心枚举/行结构。
- **设计**：
```rust
pub enum FactType { World, Experience, Opinion, Observation }
pub enum LinkType { Temporal, Semantic, Causal }
pub enum ChunkStatus { Pending, Extracting, Done, Failed }
pub enum CausalRelation { Causes, CausedBy, Enables, Prevents }   // 归一化待拍板
pub enum EntityKind { Person, Organization, Location, Product, Concept, Other }
pub enum SourceStatus { Received, Chunked }
pub enum SourceKind { Transcript, Document, Message, Api }

pub struct BankProfile { pub background: String, pub skepticism: u8, pub literalism: u8, pub empathy: u8, pub bias_strength: f32 }
impl BankProfile { pub fn validate(&self)->Result<()>; }   // 1..=5, 0..=1

pub struct Bank { id, name: String, profile: BankProfile, created_at, updated_at }
pub struct Source { id, bank_id, kind, title: Option<String>, payload_text: String, status, created_at, updated_at }
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
- **验收**：与 `memory-schema.md` §3 字段一一对应；不变量写在类型注释（observation⇒chunk_id=None；reflect opinion⇒chunk_id=None 且 confidence 必填）。
- **测试**：`BankProfile::validate` 边界矩阵（0/1/5/6）。

## P0-4 结构化输出 DTO（论文 A.5.1/A.2/A.3）

**状态**：❌

- **内容**：LLM 往返契约 + JSON Schema（schemars）。
- **设计**：
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
- **验收**：`schema_for!` 快照与论文附录字段名对齐（`where` 用 serde rename）。
- **测试**：fixtures 反序列化（含缺省字段/错误类型）。

## P0-5 错误体系

**状态**：❌

- **内容**：`engine::Error` + 重试判定。
- **设计**：`pub fn is_retryable(&self)->bool`；Store/Llm 错误经 `From` 转换；`NotFound(&'static str, String)`。
- **验收**：LLM 超时/DB busy → Retryable；schema 解析失败 → Fatal；profile 越界 → Config。
- **测试**：`is_retryable` 矩阵。

## P0-6 模型类 ports

**状态**：❌

- **内容**：`Llm/Embedder/Reranker/TemporalParser/Clock`。
- **设计**：
```rust
pub struct JsonRequest { pub system: Option<String>, pub user: String,
  pub schema_name: &'static str, pub schema: serde_json::Value,
  pub temperature: f32, pub max_tokens: u32 }
#[async_trait] pub trait Llm: Send+Sync {
  async fn complete_json<T: DeserializeOwned>(&self, req: JsonRequest)->Result<T>;
  fn model(&self)->&str;
}
#[async_trait] pub trait Embedder: Send+Sync { fn model(&self)->&str; fn dim(&self)->usize;
  async fn embed(&self, texts:&[&str])->Result<Vec<Vec<f32>>>; }
#[async_trait] pub trait Reranker: Send+Sync { async fn rerank(&self, q:&str, docs:&[&str])->Result<Vec<f32>>; }
pub struct TimeRange { pub start: Timestamp, pub end: Timestamp }
pub trait TemporalParser: Send+Sync { fn parse(&self, q:&str, now: Timestamp)->Option<TimeRange>; }
pub trait Clock: Send+Sync { fn now(&self)->Timestamp; }
```
- **验收**：`Box<dyn Llm>` 可用；schema 作为参数传入（无需泛型 schema bound）。
- **测试**：空实现编译 + dyn 调用。

## P0-7 Store port（方法集）

**状态**：❌

- **内容**：全部存储契约 + DTO。
- **设计**：
```rust
pub struct NewSource { bank_id, kind, title, payload_text }
pub struct NewChunk { bank_id, source_id, chunk_index, text, content_hash, time_start, time_end }
pub struct ClaimedChunk { pub chunk: Chunk }
pub struct NewFact { pub id: FactId, bank_id, fact_type, text, chunk_id: Option<ChunkId>,
  embedding: Option<Vec<f32>>, occurred_start/end, mentioned_at, confidence }
pub struct NewBinding { pub fact_id: FactId, pub entity_id: EntityId, pub mention_text: Option<String> }
pub struct NewLink { pub from: FactId, pub to: FactId, pub link_type: LinkType,
  pub relation: Option<CausalRelation>, pub weight: f32 }
pub struct EntityStatUpdate { pub entity_id: EntityId, pub add: bool, pub at: Option<Timestamp> }
pub struct OpinionConfidenceUpdate { pub opinion_id: FactId, pub new_confidence: f32, pub new_text: Option<String> }
pub struct ChunkCommit { pub chunk_id: ChunkId, pub facts: Vec<NewFact>, pub entities: Vec<Entity>,
  pub bindings: Vec<NewBinding>, pub links: Vec<NewLink>,
  pub entity_stats: Vec<EntityStatUpdate>, pub opinion_updates: Vec<OpinionConfidenceUpdate> }
pub struct WorkLists { pub sources_received: Vec<SourceId>, pub chunks_pending: Vec<ChunkId>,
  pub chunks_stale: Vec<ChunkId>, pub facts_missing_embedding: Vec<FactId>,
  pub entities_stale_observation: Vec<EntityId> }

#[async_trait] pub trait Store: Send+Sync {
  // bank/收件箱
  async fn create_bank(&self, name:&str, profile:BankProfile)->Result<BankId>;
  async fn load_bank(&self, id:BankId)->Result<Bank>;
  async fn create_source(&self, s:NewSource)->Result<SourceId>;
  async fn append_chunks(&self, src:SourceId, chunks:Vec<NewChunk>)->Result<usize>; // 返回新插入数
  async fn claim_chunk(&self, bank:BankId, stale_after:Duration)->Result<Option<ClaimedChunk>>;
  async fn release_chunk(&self, chunk:ChunkId)->Result<()>;
  async fn commit_chunk(&self, c:ChunkCommit)->Result<()>;
  // 检索
  async fn search_vector(&self, bank:BankId, q:&[f32], k:usize)->Result<Vec<(FactId,f32)>>;
  async fn search_bm25(&self, bank:BankId, q:&str, k:usize)->Result<Vec<(FactId,f32)>>;
  async fn neighbors(&self, bank:BankId, from:FactId)->Result<Vec<Neighbor>>;
  async fn facts_in_range(&self, bank:BankId, r:TimeRange)->Result<Vec<FactId>>;
  async fn fetch_facts(&self, ids:&[FactId])->Result<Vec<Fact>>;
  async fn increment_access(&self, ids:&[FactId])->Result<()>;
  // 实体/意见/观察/维护
  async fn ensure_entity(&self, bank:BankId, name:&str, kind:EntityKind)->Result<(EntityId,bool)>;
  async fn entity_candidates(&self, bank:BankId, mention:&str)->Result<Vec<EntityCandidate>>;
  async fn insert_opinion(&self, o:NewOpinion)->Result<FactId>;
  async fn insert_observations(&self, o:Vec<NewObservation>)->Result<()>;
  async fn observations_of(&self, bank:BankId, entity:EntityId, limit:usize)->Result<Vec<Fact>>;
  async fn pending_work(&self, bank:BankId)->Result<WorkLists>;
  async fn reembed_batch(&self, bank:BankId, limit:usize)->Result<usize>;
  async fn gc_observations(&self, bank:BankId, max_age:Duration, per_entity_cap:usize)->Result<usize>;
}
pub struct Neighbor { pub fact_id: FactId, pub link_type: LinkType, pub weight: f32 }
pub struct EntityCandidate { pub entity: Entity, pub variants: Vec<String>, pub last_mention_at: Option<Timestamp> }
```
- **关键语义**（写进 trait 文档注释）：`claim_chunk` 原子认领并回收过期 extracting；`commit_chunk` 单事务含 status→done；`append_chunks` 哈希冲突跳过；`neighbors` 合并 `fact_link` 与 `fact_entity` 推导邻接。
- **验收**：trait object-safe；DTO 无后端类型泄漏。
- **测试**：编译期 object safety 测试；DTO serde。

## P0-8 Store 形态定稿

**状态**：❌

- **内容**：单 trait 决策落地 + 拆分预案注释（读/写/维护）。
- **验收**：`let s: Box<dyn Store>` 编译；文档记录拆分触发条件（某后端只支持只读等）。
- **测试**：dyn 注入冒烟。

## P0-9 migration 0001（SQLite）

**状态**：❌

- **内容**：7 表 DDL + FTS + 索引 + 约束。
- **设计**：
  - 表按 schema 字段；FTS5：`CREATE VIRTUAL TABLE fact_fts USING fts5(fact_id UNINDEXED, text, tokenize='trigram')`（手动维护）；
  - 唯一：`chunk(bank_id, content_hash)`、`entity(bank_id, canonical_name)`、`fact_link(from_fact_id, to_fact_id, link_type, relation)`；
  - CHECK：枚举值域；`(fact_type='opinion' AND chunk_id IS NULL) => confidence IS NOT NULL`；
  - 索引：`fact(bank_id)`、`fact(chunk_id)`、`fact(bank_id, occurred_start)`、`source(bank_id,status)`、`fact_entity(bank_id, entity_id)`、`fact_entity(bank_id, fact_id)`、`fact_link(bank_id, from_fact_id)`、`fact_link(bank_id, to_fact_id)`。
- **验收**：与 `memory-schema.md` §3 逐条核对表（在 migration 文件头列 checklist）。
- **测试**：约束违规（重复哈希、重复规范名、confidence 缺失）；FTS 插入/查询。

## P0-10 migration runner + 测试库 helper

**状态**：❌

- **内容**：`store::migrate(&SqlitePool)`；`store::testing::memory_pool()`。
- **验收**：migrate 幂等（跑两次）；测试库每次独立。
- **测试**：两次 migrate；并发连接 basic。

## P0-11 fakes

**状态**：❌

- **内容**：确定性测试替身。
- **设计**：
```rust
pub struct FakeLlm { scripted: Mutex<HashMap<&'static str, VecDeque<serde_json::Value>>>, calls: Mutex<Vec<JsonRequest>> }
impl FakeLlm { pub fn push(&self, schema_name:&'static str, value:serde_json::Value); pub fn calls(&self)->Vec<JsonRequest>; }
pub struct HashEmbedder { dim: usize }        // 字符 n-gram 哈希 → 归一化向量
pub struct IdentityReranker;
pub struct RuleTemporalParser;                 // "yesterday"/"last week"/ISO 日期
pub struct FixedClock(pub Timestamp);
```
- **验收**：同输入同输出；`FakeLlm` 可断言 prompt 内容（画像口头化测试用）。
- **测试**：嵌入确定性/维度；FakeLlm 队列耗尽返回 Fatal。

## P0-12 P0 冒烟

**状态**：❌

- **内容**：bank→source→chunks→claim/release 集成测试。
- **验收**：全绿；`cargo doc` 无 broken link。
- **测试**：`crates/store/tests/smoke.rs`。

---

# P1 · retain

## P1-1 载荷规范化

**状态**：❌

- **内容**：把调用方输入转成规范文本 + 轮次视图。
- **设计**：
```rust
pub struct RawTurn { pub speaker: String, pub text: String, pub at: Option<Timestamp>, pub session: Option<u32> }
pub enum RawPayload { Transcript { title: Option<String>, turns: Vec<RawTurn> }, Document { title: Option<String>, text: String } }
pub struct TurnSpan { pub byte_start: usize, pub byte_end: usize, pub at: Option<Timestamp>, pub session: Option<u32> }
pub struct NormalizedPayload { pub text: String, pub spans: Vec<TurnSpan> }
pub fn normalize(p: RawPayload) -> NormalizedPayload   // "[<ts>] <speaker>: <text>\n"
```
- **验收**：Document 直通；Transcript 保留时间/会话标记；span 可逆（切片==轮次文本）。
- **测试**：空轮次、CJK、超长单轮。

## P1-2 分块器

**状态**：❌

- **内容**：会话对齐 + 窗口兜底 + 哈希。
- **设计**：
```rust
pub struct ChunkConfig { pub session_gap: Duration, pub max_tokens: usize, pub min_tokens: usize, pub estimator: TokenEstimatorKind }
pub struct RawChunk { pub index: u32, pub text: String, pub content_hash: String,
                      pub time_start: Option<Timestamp>, pub time_end: Option<Timestamp> }
pub fn chunk(p: &NormalizedPayload, cfg: &ChunkConfig, est: &dyn TokenEstimator) -> Vec<RawChunk>
pub fn chunk_hash(text: &str) -> String   // 空白归一 + SHA-256
```
- **验收**：确定性（同输入同输出）；不产生空块；超长会话按窗口切且不跨"明显会话边界"。
- **测试**：多会话、单超长轮、仅空白差异哈希相等。

## P1-3 `create_source`（tx0）

**状态**：❌

- **内容**：Store 实现 + 事务边界。
- **验收**：payload_text 落库、status=received；崩溃点前无半写。
- **测试**：读回一致；重复 external 无意义 ID 均新行。

## P1-4 `append_chunks`（tx1）

**状态**：❌

- **内容**：批量插入 + 冲突跳过 + source 状态推进。
- **设计**：`INSERT … ON CONFLICT(bank_id,content_hash) DO NOTHING`；同事务 `UPDATE source SET status='chunked'`。
- **验收**：全冲突时插入 0 行但 source 照常 chunked（零 chunk 账本行）。
- **测试**：同源重灌两次；跨源相同内容；部分冲突。

## P1-5 `claim_chunk`

**状态**：❌

- **内容**：原子认领 + 过期回收。
- **设计**：
```sql
UPDATE chunk SET status='extracting', updated_at = ?1
WHERE id = (SELECT id FROM chunk WHERE bank_id=?2 AND status='pending'
            ORDER BY created_at LIMIT 1)
  AND status='pending'
RETURNING *;
-- 过期回收：UPDATE chunk SET status='pending' WHERE status='extracting' AND updated_at < ?old
```
- **验收**：双 worker 并发只有一个拿到；过期 extracting 被重置。
- **测试**：并发 10 任务仅 1 成功；时间回拨后重置。

## P1-6 `release_chunk`

**状态**：❌

- **内容**：失败回 pending。
- **验收**：status=pending、updated_at 刷新；连续失败不改变语义。
- **测试**：release 后可再次 claim。

## P1-7 抽取调用

**状态**：❌

- **内容**：A.1 prompt + schema + 解析校验。
- **设计**：
```rust
pub async fn extract_facts(llm:&dyn Llm, chunk:&Chunk, cfg:&ExtractionConfig) -> Result<ExtractionReport>
pub struct ExtractionReport { pub facts: Vec<ExtractedFact>, pub discarded: usize }
// prompt: A.1 全文（system），user=chunk.text；schema_name="fact_extraction"
// 校验：what 非空；fact_type 合法；causal.target_fact_index < facts.len()
pub fn validate_extracted(f: &ExtractedFact) -> Result<()>
```
- **验收**：单条不合法丢弃并计数；整调用重试仅对 Retryable。
- **测试**：FakeLlm 返回 3 条含 1 条坏；空结果；超长文本截断策略。

## P1-8 时间与增强文本

**状态**：❌

- **内容**：时间归一化 + 人类可读前缀 + 嵌入输入组装。
- **设计**：
```rust
pub fn normalize_time(extracted:&mut ExtractedFact, chunk:&Chunk, clock:&dyn Clock) -> Result<()>
// 缺失 occurred → 用 chunk.time_start/end；缺失 mentioned_at → chunk 区间中点/创建时刻
pub fn human_readable(ts: Timestamp) -> String            // "Saturday, June 9, 2024"
pub fn embedding_input(text:&str, ts: Option<Timestamp>) -> String  // "{human}. {text}"
```
- **验收**：无时间信息时不伪造（None 保持 None）；存库文本 == 嵌入输入（同一字符串）。
- **测试**：相对时间已由 LLM 归一（本步只做缺省补全）；前缀格式。

## P1-9 实体消解 ρ

**状态**：❌

- **内容**：精确命中 → 三特征模糊 → ensure_entity。
- **设计**：
```rust
pub struct ResolveWeights { pub alpha: f32, pub beta: f32, pub gamma: f32, pub threshold: f32 }
pub struct MentionContext { pub co_mentioned: Vec<EntityId>, pub at: Option<Timestamp> }
pub async fn resolve(store:&dyn Store, bank:BankId, mention:&EntityMention,
                     ctx:&MentionContext, w:&ResolveWeights) -> Result<EntityId>
pub fn normalize_name(s:&str)->String                    // lowercase/trim/空白折叠
pub fn string_sim(a:&str,b:&str)->f32                    // 归一化 Levenshtein 相似
pub fn cooccurrence_sim(cand:&EntityCandidate, ctx:&MentionContext)->f32
pub fn temporal_sim(cand:&EntityCandidate, at: Option<Timestamp>)->f32
```
- **验收**：同 mention 永远解析到同一实体；冲突创建用 `ensure_entity` 的 UNIQUE 兜底。
- **测试**：精确/变体/新实体三路；并发建同名实体只产生一行；空 mention 拒绝。

## P1-10 实体统计

**状态**：❌

- **内容**：commit 内应用 `EntityStatUpdate`。
- **设计**：`mention_count += 1`；`first_seen = min`；`last_seen = max`（对话时钟）。
- **验收**：与绑定行数一致（测试断言 COUNT）。
- **测试**：新增/重抽替换（-1/+1）；NULL 时间不更新 seen。

## P1-11 嵌入生成

**状态**：❌

- **内容**：批量嵌入 + 维度校验 + 失败降级。
- **设计**：`pub async fn embed_facts(emb:&dyn Embedder, texts:&[String], batch:usize) -> Vec<Option<Vec<f32>>>`；校验 `len==dim()`；失败记 None（运行时回填）。
- **验收**：维度不一致直接 Config 错；失败不阻塞 tx2。
- **测试**：批次切分；单批失败 → None 行。

## P1-12 temporal 建链

**状态**：❌

- **内容**：区间邻近 → 双行。
- **设计**：
```rust
pub fn temporal_weight(gap: Duration, sigma: Duration) -> f32   // exp(-gap/sigma)
pub async fn build_temporal_links(store:&dyn Store, bank:BankId, new_facts:&[Fact],
                                  cfg:&LinkConfig) -> Result<Vec<NewLink>>
// 候选：与新区间距离 < 窗口；按权重排序取前 K 对；每对写 from→to 与 to→from
```
- **验收**：双行成对（规范键计数恒 2）；按对裁剪不留单侧；无时间行不建边。
- **测试**：三事实时间窗；裁剪边界；权重公式单测。

## P1-13 semantic 建链

**状态**：❌

- **内容**：有向 top-k。
- **设计**：`pub async fn build_semantic_links(store, bank, new_facts, cfg) -> Result<Vec<NewLink>>`；对每个新 fact：`search_vector`（排除自身、cos≥θs）取 top-K，每条 `from=new, to=cand, weight=cos`。
- **验收**：单行有向；不建 B→A（除非 B 也选 A）；预算内。
- **测试**：预算 2 时同分布断言；自身排除；θs 过滤。

## P1-14 causal 建链

**状态**：❌

- **内容**：抽取输出映射。
- **设计**：`pub fn causal_links(extracted:&[ExtractedFact], ids:&[FactId]) -> Vec<NewLink>`；`target_fact_index → ids[i]`；单行。归一化按拍板结果（当前：原样 + relation）。
- **验收**：越界索引丢弃；重复对去重；强度不入权重（论文 w=1.0）。
- **测试**：两事实单向因果；链式因果；越界。

## P1-15 意见强化

**状态**：❌

- **内容**：候选 → Assess → 置信度更新。
- **设计**：
```rust
pub struct ReinforcementConfig { pub alpha: f32, pub sim_threshold: f32, pub max_candidates: usize }
pub async fn reinforce(store:&dyn Store, llm:&dyn Llm, bank:BankId,
                       new_facts:&[Fact], cfg:&ReinforcementConfig)
  -> Result<Vec<OpinionConfidenceUpdate>>
// 候选：实体重叠 ∪ sim(v_o,v_f)>θ（search_vector 限 opinion）
// Assess schema：{"verdict":"reinforce|weaken|contradict|neutral","reason":"…"}
// c' = min(c+α,1) / max(c-α,0) / max(c-2α,0) / c
```
- **验收**：四档规则单测；neutral 不产生更新；contradict 可选改文本；全部随 tx2 原子。
- **测试**：FakeLlm 四 verdict；实体重叠/向量召回两路候选；α 边界(0/1)。

## P1-16 `commit_chunk`（tx2）

**状态**：❌

- **内容**：单事务落全部产物。
- **设计**：顺序 insert entity→fact→fact_entity→fact_link→FTS→stat updates→opinion updates→chunk.status=done；任一失败回滚。
- **验收**：失败后无任何残留行；成功后可观察到 fact/边/统计一致。
- **测试**：注入失败（如 FTS 唯一冲突）验证全回滚；重放同一 commit 幂等（status 已 done 时拒绝）。

## P1-17 retain API（同步语义）

**状态**：❌

- **内容**：facade 方法。
- **设计**：
```rust
pub struct RetainOptions { pub wait_until_done: bool }   // v1 恒 true
pub struct RetainReport { pub source_id: SourceId, pub chunks_total: usize,
                          pub chunks_skipped: usize, pub facts_written: usize }
impl MemoryBank { pub async fn retain(&self, bank:BankId, payload:RawPayload, opts:RetainOptions) -> Result<RetainReport> }
```
- **验收**：返回时本次 chunks 全部 done（或被跳过）；错误分类传播（Retryable 包装 context）。
- **测试**：端到端（fake LLM/embedder）产出事实数正确。

## P1-18 worker + rescan

**状态**：❌

- **内容**：派生 work list 驱动。
- **设计**：
```rust
pub struct RescanReport { pub sources_chunked: usize, pub chunks_processed: usize, pub failed: usize }
impl MemoryBank { pub async fn rescan(&self, bank:BankId) -> Result<RescanReport>;
                  pub fn spawn_worker(&self, cfg:WorkerConfig) -> JoinHandle<()>; }
```
- **验收**：status=received 补分块；pending/过期 extracting 重跑；failed 不自动重试（运维重置）。
- **测试**：中断模拟（手工置 status）后 rescan 收敛。

## P1-19 retain 集成测试

**状态**：❌

- **内容**：端到端矩阵。
- **验收/测试**：
  - 重灌幂等（零新 facts）；增量重灌（只处理新 chunks）；
  - 抽集中断 → rescan 恢复；tx2 失败 → 无半成品；
  - 双行不变量（SQL 断言计数=2）；实体统计与绑定一致。

---

# P2 · recall

## P2-1 语义通道

**状态**：❌

- **设计**：`search_vector` 实现：`SELECT id, embedding FROM fact WHERE bank_id=? AND embedding IS NOT NULL`，Rust 内余弦，维护 top-k 最小堆；`k`/`min_cos` 配置。
- **验收**：与暴力参考实现一致；NULL 跳过。
- **测试**：已知向量集排序断言；空库返回空。

## P2-2 BM25 通道

**状态**：❌

- **设计**：`search_bm25`：查询词按空白/标点切分并转义（`"term"` 引号包裹），`SELECT fact_id, bm25(fact_fts) AS s … JOIN fact ON … WHERE bank_id=? ORDER BY s LIMIT k`。
- **验收**：中文 trigram 可命中子串；空查询返回空。
- **测试**：专名/数字命中；注入式特殊字符不报错。

## P2-3 图通道（扩散激活）

**状态**：❌

- **设计**：
```rust
pub struct GraphConfig { pub delta: f32, pub mu_temporal/mu_semantic/mu_entity/mu_causal: f32,
                         pub max_depth: u8, pub max_visits: usize }
pub async fn spread(store:&dyn Store, bank:BankId, seeds:&[(FactId,f32)], cfg:&GraphConfig)
  -> Result<Vec<(FactId,f32)>>   // A(fj)=max(A(fi)*w*delta*mu(ℓ))
```
- **验收**：深度/访问上限生效；确定性（同输入同序）。
- **测试**：链式图激活衰减；环不无限；实体虚拟邻接参与。

## P2-4 时间通道

**状态**：❌

- **设计**：`TemporalParser.parse` → `facts_in_range` → 邻近分 `1 - |midDiff|/(range/2)`。
- **验收**：无时间约束时不触发；区间端点闭开语义固定（闭区间）。
- **测试**：查询"上周"；无结果；跨零时区统一 UTC。

## P2-5 RRF

**状态**：❌

- **设计**：`pub fn rrf(lists:&[Vec<FactId>], k:u32)->Vec<(FactId,f32)>`；并集、稳定排序（score desc, id asc）。
- **验收**：单列表=原序；缺失列表贡献 0。
- **测试**：两列表交叉；平分稳定性。

## P2-6 rerank 接入

**状态**：❌

- **设计**：`pub async fn rerank_top(reranker:&dyn Reranker, query:&str, facts:&[Fact], top_n:usize)`；出错回退原序并记录。
- **验收**：只重排 top-N；其余保持。
- **测试**：Identity 保持；反转 fake 生效。

## P2-7 预算打包

**状态**：❌

- **设计**：
```rust
pub trait TokenEstimator { fn estimate(&self, text:&str)->usize }
pub struct CharHeuristic;      // ceil(chars * ratio) 可配
pub fn pack(facts:&[(Fact,f32)], budget:usize, est:&dyn TokenEstimator) -> (Vec<Fact>, usize)
```
- **验收**：累计 ≤ budget；超界即停；首条超预算时固定返回空并告警。
- **测试**：边界 budget=0/恰好等于/差1。

## P2-8 召回产物与 access_count

**状态**：❌

- **设计**：
```rust
pub struct RecalledFact { pub fact: Fact, pub score: f32, pub channels: Vec<(Channel, u32 /*rank*/)> }
pub struct RecallResult { pub facts: Vec<RecalledFact>, pub tokens_used: usize }
```
- **验收**：channels 标注重合；`increment_access` 批量执行。
- **测试**：多通道命中标注。

## P2-9 recall API

**状态**：❌

- **设计**：
```rust
pub struct RecallOptions { pub token_budget: usize, pub channels: ChannelMask,
                           pub time_range: Option<TimeRange>, pub rerank_top_n: usize, pub per_channel_k: usize }
impl MemoryBank { pub async fn recall(&self, bank:BankId, query:&str, opts:RecallOptions) -> Result<RecallResult> }
```
- **验收**：四通道可独立关闭；query 嵌入仅一次；总耗时打点。
- **测试**：固定语料 10 问的命中断言（fake embedder 语义可构造）。

## P2-10 性能基线

**状态**：❌

- **设计**：`benches/`（criterion）或 example 计时；记录 1k/10k 暴力余弦耗时。
- **验收**：数据入库 `design/` 备查（不设硬阈值）。
- **测试**：基准可运行（CI 不跑）。

---

# P3 · reflect

## P3-1 φ(Θ) 口头化

**状态**：❌

- **设计**：`pub fn verbalize(p:&BankProfile)->String`；S/L/E 各 5 档文案 + β 三档修饰（0/0.5/1 规则）。
- **验收**：同画像输出稳定；β=0 时弱化偏好措辞。
- **测试**：极值组合快照。

## P3-2 system message 组装

**状态**：❌

- **设计**：`pub fn system_message(bank:&Bank, evidence:&[RecalledFact])->String`（姓名/背景/φ/证据编号块）。
- **验收**：证据按编号可引用；无证据时显式"无相关记忆"。
- **测试**：快照；超长证据截断策略。

## P3-3 回答 + 意见抽取

**状态**：❌

- **设计**：`pub async fn reflect_llm(llm:&dyn Llm, sys:String, user:String)->Result<ReflectResponse>`（schema `reflect`）。
- **验收**：意见 confidence ∈[0,1]；空 opinions 合法。
- **测试**：FakeLlm 返回含 2 条意见。

## P3-4 意见落库

**状态**：❌

- **设计**：`NewOpinion { id, bank_id, text, confidence, entities: Vec<(EntityId, Option<String>)>, embedding }`；实体绑定：`bind_known_entities(store, bank, text)`（对 canonical_name/变体做精确/子串匹配，不调 LLM）。
- **验收**：chunk_id=None；confidence 必填；嵌入写入。
- **测试**：绑定 0/2 实体；文本含多个已知实体。

## P3-5 强化复用

**状态**：❌

- **设计**：reflect 后对新意见与既有意见调用同一 `reinforce()` 入口。
- **验收**：无逻辑分叉（同一实现）。
- **测试**：P1-15 用例复用。

## P3-6 证据透传

**状态**：❌

- **设计**：`ReflectResult { answer, opinions_formed: Vec<FactId>, recalled: RecallResult }`。
- **验收**：回答与证据 id 绑定关系可外查（不解析文本）。
- **测试**：返回结构断言。

## P3-7 行为测试

**状态**：❌

- **设计**：FakeLlm 记录请求；断言高/低怀疑度画像的 system prompt 差异与 β 调制。
- **验收**：prompt 断言稳定（快照）。
- **测试**：两组画像对比。

## P3-8 reflect API

**状态**：❌

- **设计**：`impl MemoryBank { pub async fn reflect(&self, bank, query, opts:ReflectOptions)->Result<ReflectResult> }`（内部 recall + LLM + 落库）。
- **验收**：端到端跑通；失败可重试（意见 id 预生成 + 单事务插入，避免重复）。
- **测试**：集成（fake）。

---

# P4 · observe 与打磨

## P4-1 观察合成调用

**状态**：❌

- **设计**：`pub async fn synthesize_observations(llm, entity, facts, cfg) -> Result<Vec<ObservationDraft>>`（A.3，3–7 条，客观第三人称）。
- **验收**：条数范围；空事实集不调用。
- **测试**：FakeLlm 条数/内容校验。

## P4-2 证据变更 work list

**状态**：❌

- **设计**：`pending_work.entities_stale_observation`：`entity.last_seen > max(observation.created_at bound to entity)`（SQL 聚合 + 索引）。
- **验收**：新证据触发；无变更不重复触发；上限参数。
- **测试**：新增事实→实体入列；再合成后出列。

## P4-3 追加式插入

**状态**：❌

- **设计**：`NewObservation { fact_id, bank_id, text, embedding, entity_ids, created_at }`；一次事务插入行 + 全量绑定 + FTS。
- **验收**：多实体绑定；无 chunk_id；旧批不删。
- **测试**：重复合成产生新行（非替换）。

## P4-4 观察取用

**状态**：❌

- **设计**：`observations_of` 按 `created_at DESC LIMIT n`；facade 暴露 `entity_profile(bank, entity, limit)`。
- **验收**：新近度优先；与 recall 自然检索并存。
- **测试**：两批观察的顺序断言。

## P4-5 GC

**状态**：❌

- **设计**：`gc_observations(bank, max_age, per_entity_cap)`：先删超龄，再按实体绑定计数保留最新 cap 条；同事务。
- **验收**：不影响非 observation 行；幂等。
- **测试**：龄/计数两种裁剪；共享实体行被多实体计数。

## P4-6 sweeper

**状态**：❌

- **设计**：`spawn_sweeper(cfg)`：周期执行 rescan + 观察合成队列 + 嵌入回填 + GC；去抖（同实体在跑跳过）；jitter。
- **验收**：崩溃后重启自动收敛；不重复合成同批。
- **测试**：两轮 sweep 后工作清空。

## P4-7 质量打磨

**状态**：❌

- **设计**：消解阈值/共现权重、建链剪枝、分块参数全部配置化；加数据驱动的调参脚本（examples）。
- **验收**：配置变更无需改代码。
- **测试**：配置快照/默认值。

## P4-8 P4 集成测试

**状态**：❌

- **测试**：观察追加/回收、GC 与 recall 交互、实体画像 API。

---

# P5 · 适配器与评测

## P5-1 PG migration + 适配器

**状态**：❌

- **设计**：独立 `postgres` module（sqlx::Postgres），类型映射 jsonb/timestamptz；`vector` 类型可选 feature；SQL 方言差异隔离在 module 内。
- **验收**：同一 trait 双实现；契约测试通过。
- **测试**：testcontainers 或本地 PG（CI 可选）。

## P5-2 Store 契约测试双跑

**状态**：❌

- **设计**：`store::testing::contract(factory: impl BackendFactory)`，覆盖方法语义（认领原子性、冲突跳过、邻接合并、统计维护）。
- **验收**：两后端全绿。
- **测试**：契约套件本身。

## P5-3 真 reranker

**状态**：❌

- **设计**：`llm::HttpReranker { endpoint, model, timeout }`；批量文档打分；失败回退原序。
- **验收**：与 stub 可互换。
- **测试**：mock HTTP（wiremock）。

## P5-4 真 TemporalParser

**状态**：❌

- **设计**：规则库（多语言日期/相对表达）+ 可选 t5-small 兜底（HTTP/ONNX 二选一）。
- **验收**：常见表达解析率；无法解析返回 None。
- **测试**：日期语料表驱动。

## P5-5 并发与配置化收口

**状态**：❌

- **设计**：worker 并发、批次、超时/重试、日志采样全部配置；tracing spans（bank/chunk/fact）。
- **验收**：压测下无锁争用热点；重试指标可见。
- **测试**：配置组合冒烟。

## P5-6 LongMemEval harness

**状态**：❌

- **设计**：数据加载 → 按 session retain → 逐问 recall+reflect → LLM-as-judge（论文 A.4 prompt）→ 指标汇总。
- **验收**：可复现跑通（fake 模式先验证管线）。
- **测试**：小样本 golden。

## P5-7 LoCoMo harness

**状态**：❌

- **设计**：多会话对话 + 图像字段忽略策略 → 同 P5-6。
- **验收**：指标输出格式统一。
- **测试**：小样本。

## P5-8 文档与示例

**状态**：❌

- **设计**：README 快速开始；examples：`retain_basic`、`recall_query`、`reflect_with_profile`、`custom_store`（展示 port 可替换）。
- **验收**：`cargo run --example` 可跑（fake 模式无 API key）。
- **测试**：examples 编译 + 冒烟。

---

# 交付节奏说明

- 每步一提交，提交信息 `P<阶段>-<序号>: <任务>`；
- P0/P1 是主干，建议**逐任务评审后合并**；P2 起可两条线并行（检索/意见）；
- 每阶段末跑：`cargo test --workspace` + 阶段集成测试 + clippy/fmt；
- 未决语义在对应任务开工前拍板：§7.A#2（confidence）、causal 归一化/relation 收窄、Store 是否拆分、token 估算器实现。
