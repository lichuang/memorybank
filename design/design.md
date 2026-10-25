# memorybank 实现设计（Rust）

依据：`../docs/hindsight.pdf`（论文）与 `memory-schema.md`（逻辑表结构定稿，7 张表 + §1.3 准入判据）。

## 0. 工程形态

Workspace 布局（crate 名不带产品前缀；`core` 因与 Rust 内建 crate 同名而回避）：

```
memorybank/                    # workspace 根；根 package = facade
├── crates/
│   ├── engine/                # 领域类型 + ports + 三操作编排（核心，无具体后端依赖）
│   │   ├── types/             # Bank, Source, Chunk, Fact, Entity, FactEntity, FactLink...
│   │   ├── ports/             # Store, Llm, Embedder, Reranker, TemporalParser, Clock
│   │   ├── retain/            # 收件箱、分块、抽取编排、tx2 提交、rescan
│   │   ├── recall/            # 四通道 + RRF + rerank + 预算打包
│   │   ├── reflect/           # 回答生成 + 意见形成 + 强化
│   │   ├── observe/           # 观察合成 + GC（work list 推导）
│   │   ├── resolve/           # 实体消解 ρ
│   │   └── link/              # temporal / semantic / causal 建链
│   ├── store/                 # sqlx：SQLite（默认）/ PostgreSQL（feature gate）
│   └── llm/                   # OpenAI-compatible chat / embedding provider
└── examples/, tests/
```

| crate | 职责 | 依赖 |
|---|---|---|
| `memorybank`（根） | facade：组装 ports + 对外 API + builder；用户唯一直接依赖 | engine + store + llm |
| `engine` | 领域类型、端口定义、retain/recall/reflect 编排 | 无具体后端/模型 |
| `store` | 实现 `Store`（SQLite/PG；索引与查询方言在此收敛） | engine（ports） |
| `llm` | 实现 `Llm` / `Embedder`（OpenAI-compatible HTTP） | engine（ports） |

依赖方向固定为 **facade → engine ← adapters**；engine 不反向依赖任何适配器。用户可自实现任意 port（自定义存储、本地模型）替换默认适配器。

## 1. Ports（接口契约）

```rust
#[async_trait]
pub trait Llm: Send + Sync {
    /// 唯一入口：结构化输出。抽取(A.1/A.5.1)、意见形成(A.2)、
    /// 观察合成(A.3)、Assess、背景合并都复用；prompt 与 JSON schema 在 engine 构造。
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
    fn now(&self) -> DateTime<Utc>; // 测试可注入
}

#[async_trait]
pub trait Store: Send + Sync {
    // —— 收件箱（schema §5.1 四段）——
    async fn create_source(&self, new: NewSource) -> Result<SourceId>;                        // tx0
    async fn append_chunks(&self, src: SourceId, chunks: Vec<NewChunk>) -> Result<usize>;    // tx1；哈希冲突跳过
    async fn claim_chunk(&self, bank: BankId) -> Result<Option<ClaimedChunk>>;               // pending→extracting 原子认领
    async fn commit_chunk(&self, commit: ChunkCommit) -> Result<()>;                         // tx2：全部产物 + status=done 原子提交
    async fn release_chunk(&self, chunk: ChunkId) -> Result<()>;                             // 失败回 pending

    // —— 读路径（schema §5.4 四通道）——
    async fn search_vector(&self, bank: BankId, q: &[f32], k: usize) -> Result<Vec<(FactId, f32)>>;
    async fn search_bm25(&self, bank: BankId, q: &str, k: usize) -> Result<Vec<(FactId, f32)>>;
    async fn neighbors(&self, bank: BankId, from: FactId) -> Result<Vec<Neighbor>>;          // fact_link + fact_entity 推导合并
    async fn facts_in_range(&self, bank: BankId, range: TimeRange) -> Result<Vec<FactId>>;
    async fn fetch_facts(&self, ids: &[FactId]) -> Result<Vec<Fact>>;

    // —— 实体 / 意见 / 观察 / 维护 ——
    async fn upsert_entity(&self, new: NewEntity) -> Result<EntityId>;                       // 含 UNIQUE(bank_id, canonical_name) 冲突处理
    async fn entity_candidates(&self, bank: BankId, mention: &str) -> Result<Vec<EntityCandidate>>;
    async fn insert_opinion(&self, op: NewOpinion) -> Result<FactId>;
    async fn update_opinion(&self, id: FactId, confidence: f32, text: Option<String>) -> Result<()>;
    async fn insert_observations(&self, obs: Vec<NewObservation>) -> Result<()>;
    async fn observations_of(&self, bank: BankId, entity: EntityId, limit: usize) -> Result<Vec<Fact>>;
    async fn pending_work(&self, bank: BankId) -> Result<WorkLists>;                        // 派生 work list
    async fn reembed_batch(&self, bank: BankId, limit: usize) -> Result<usize>;
    async fn stats(&self, ...) -> ...;
}
```

刻意的契约取舍（均对应 schema §1.3 判据）：

1. **`commit_chunk` 是唯一事务入口**：原子性由 trait 方法保证，事务对象不外泄；engine 不需要理解各后端的事务 API。
2. **`neighbors()` 由适配器合并多源**：`fact_link` 行 + `fact_entity` 推导的实体邻接（闭包不物化）——判据 3“单一事实源”。
3. **work list 全部派生**：`pending_work` 返回待分块 source、待抽取/过期认领 chunk、NULL embedding、证据变更实体（`entity.last_seen > 最新观察行 created_at`）；无任务队列表——判据 2“策略归运行时”。
4. **无模型/阈值字段**：embedding 模型、分块参数、σt/θs/top-k、µ(ℓ) 全在配置——判据 2。
5. **`search_vector` 必须可用**：ANN 是适配器内部优化（SQLite 暴力余弦；PG 可后加 pgvector 索引），契约只要求“返回 top-k”。

## 2. 管线实现要点

### 2.1 retain（收件箱 + 抽取）

```
ingest(payload):
  规范化文本 → tx0: create_source
  分块（会话对齐优先/窗口兜底）→ tx1: append_chunks（哈希冲突跳过）
  [同步模式] 立即处理本次 pending chunks；[异步模式] 交给 worker
worker loop:
  claim_chunk → 抽取(Llm, A.1/A.5.1) → 消解(Store) → 嵌入(Embedder)
             → 建链(link 模块) → 强化(Assess + 置信度增量)
             → commit_chunk(tx2)   // 失败: release_chunk，回 pending
rescan(): pending_work → 补分块 / 重抽取（过期 extracting 用 updated_at 龄判定）
```

- 多模型调用（抽取/Assess/embedding）全部在 tx2 **之外**；tx2 内为纯 DB 写；
- 初版默认**同步**（`retain` 阻塞至本次 chunk 全 done，schema §7.B#8），worker/rescan 作为独立 API。

### 2.2 recall（四通道 → RRF → rerank → 预算）

- `search_vector` / `search_bm25` 与图扩展、时间过滤并发（`try_join!`）；
- RRF：`Σ 1/(60 + rank_i)`，按 fact_id 聚合；
- rerank：`Reranker` 对 top-N 候选打分（初版恒等 stub）；
- 预算打包：按最终序贪心累加 token（估算器可插拔），超预算截断；命中行 `access_count++`。

### 2.3 reflect（回答 + 意见形成 + 强化）

- recall 取证据 → 组装系统消息（姓名 n、背景 h、φ(Θ) 口头化 + β 强弱的措辞调制）；
- 一次 `Llm::complete_json` 产出 { 回答文本, 候选意见[] }（A.2：第一人称、confidence、reasoning）；
- 新意见 `insert_opinion`；对既有意见做强化（Assess → c±α，强矛盾可改写文本）。

### 2.4 observe（观察合成 + GC）

- work list：证据变更的实体（`entity.last_seen` 晚于其最新观察行）→ `Llm` 合成 3–7 条客观陈述（A.3）→ 追加式 `insert_observations`（全量绑定）；
- GC：按龄 / 每实体绑定计数上限（初版可不做）；
- 调度：运行时 sweeper（无持久队列）。

### 2.5 resolve（实体消解）

- 规范化 mention → 先精确命中 `entity.canonical_name`（索引直达）；未命中走 ρ 三特征（string / co-occurrence / temporal），变体从 `fact_entity.mention_text` 派生（调用内缓存，不落缓存表）；
- 落库：`upsert_entity`（UNIQUE 冲突即复用）+ 绑定行 + 实体统计增量（mention_count / first_seen / last_seen，同事务）。

### 2.6 link（建链）

- **temporal**：区间距离 → `exp(−Δt/σt)`；窗口内按对裁剪，成对写入双行；
- **semantic**：嵌入算好后 ANN top-k（预算谓词），**有向单行**（from=选择方）；
- **causal**：抽取输出的 `causal_relations[]`（仅同 chunk），归一为“因→果”单行；
- 建链阈值与上限全部来自配置。

## 3. 横切关注点

| 关注点 | 方案 |
|---|---|
| 异步运行时 | tokio；端口用 `async_trait`（dyn 兼容，运行时可注入） |
| 错误 | `thiserror` 分三类：`Retryable`（LLM 超时、DB 竞争）/`Fatal`（数据损坏）/`Config`；worker 仅重试 Retryable |
| 配置 | serde 结构体：bank 画像（Θ/β）、模型选择、阈值、分块参数、并发度 |
| 观测 | `tracing` 打点：每 chunk 耗时、LLM 调用次数/重试、召回各通道命中、预算使用率 |
| ID/时间 | UUIDv7；chrono UTC |
| 向量 | 内存 `Vec<f32>`；写入校验维度；适配器负责持久化形态 |
| 并发 | chunk worker 数可配；同一 chunk 由认领保证单执行；bank 内无全局锁 |

## 4. 测试策略

1. **确定性 fakes**：`FakeLlm`（按输入返回固定结构化输出）、`HashEmbedder`（字符哈希向量，确定性）、`IdentityReranker`——管线端到端可测；
2. **SQLite 真库**：每个测试独立内存库 + migration；覆盖幂等（重灌/增量/崩溃恢复）、事务回滚、双行不变量；
3. **契约测试**：`Store` 的同一套行为测试对 SQLite/PG 双跑（`store` crate 的 test kit）；
4. **评测（后置）**：LongMemEval / LoCoMo harness 接真模型（P5）。

## 5. 推进顺序

| 阶段 | 内容 | 出口标准 |
|---|---|---|
| **P0** | types + ports + SQLite migration（schema 物理落地）+ fakes | 编译通过；契约测试骨架可跑 |
| **P1** | retain 闭环：收件箱→分块→抽取→tx2→rescan | 重灌/中断恢复测试全绿 |
| **P2** | recall 闭环：四通道 + RRF + rerank stub + 预算 | 固定语料命中率符合预期 |
| **P3** | reflect：回答 + 意见形成 + 强化 | 画像调制可见；意见可演化 |
| **P4** | observe + resolve/link 打磨 | 观察可合成/回收；消解质量可评测 |
| **P5** | PG/pgvector 适配、真 reranker、TemporalParser、评测 | 双后端契约测试通过；基准可跑 |

## 6. 待拍板

1. `Store` 形态：单 trait（当前草案）vs 按能力拆分子 trait（读/写/维护）——P0 定稿；
2. retain 默认同步还是异步 API（schema §7.B#8 建议同步为首发）；
3. reranker / temporal parser 的首发实现（恒等 stub + 规则解析，模型版后置）；
4. 分块 token 化与预算估算器（是否直接复用嵌入模型 tokenizer）；
5. 评测接入时机与 harness 形态（P5）；
6. 继承自 schema 文档的语义挂账：`§7.A#2`（抽取产 opinion 行 confidence 可空 vs 初始置信度）。
