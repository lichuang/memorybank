# AGENTS.md

Guidance for AI coding agents working in this repository. Humans are welcome to follow it too.

## 1. What this project is

`memorybank` — a Rust implementation of the agent memory system from the paper **Hindsight Is 20/20: Building Agent Memory that Retains, Recalls, and Reflects** (`docs/hindsight.pdf`): four memory networks (world / experience / opinion / observation), three operations (retain / recall / reflect), TEMPR retrieval and CARA reflection.

**Status**: the design phase is complete; implementation has not started (P0 pending).

Design docs (read in this order):

| Document | Purpose |
|---|---|
| `design/phase.md` | **What to do**: 66 implementation tasks (P0–P5) with content / design / acceptance / tests and status markers |
| `design/design.md` | **How to build it**: workspace layout, port contracts, pipeline implementation, testing strategy |
| `design/memory-schema.md` | **What the data looks like**: 7 tables, admission criteria (§1.3), data flow, retired constructs (§7.C) |
| `docs/hindsight.pdf` | The paper — the final authority for design trade-offs |

## 2. Repository layout (target)

```
memorybank/                 # workspace root (virtual manifest; no root package)
├── crates/
│   ├── memorybank/         # facade: public API + wiring
│   ├── engine/             # domain types + ports + retain/recall/reflect orchestration
│   ├── store/              # Store implementation (SQLite by default, PostgreSQL later)
│   └── llm/                # Llm / Embedder providers (HTTP-compatible + optional local adapters)
├── design/                 # design documents (three files)
└── docs/                   # the paper
```

Dependency direction is fixed: **facade → engine ← adapters**. `engine` must not depend on `store` or `llm`.

> Note: P0-1 has created the crates and the virtual workspace.

## 3. Commands

```bash
cargo check --workspace
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p engine            # single crate
cargo doc --workspace --no-deps
```

- Unit and integration tests need **no external database** (in-memory SQLite) and **no API keys** (fakes are the default).
- Real providers are used only when explicitly configured (P5 and later).
- Migrations live in `crates/store/migrations/` and are applied by `store::migrate()`.

## 4. Workflow

1. **One task at a time**, taken from `design/phase.md` in phase order (P0 first).
2. Read the task's Content / Design / Acceptance / Tests sections first. If the design must change, update the design docs **before** the code (see §6).
3. When done:
   - change that task's `**Status**: ❌` to `**Status**: ✅` (add a date or commit hash if useful);
   - update the phase progress counters at the top of `design/phase.md`.
4. Commit message format: `P<phase>-<index>: <task name>`, e.g. `P0-2: id and time primitives`.
5. Every task ships with tests; run phase-level integration tests plus `fmt` / `clippy` at the end of each phase.
6. **Do not decide open design questions on your own** (listed in `design/design.md` §6 and `memory-schema.md` §7 — e.g. §7.A#2 confidence, causal normalization / relation narrowing). Ask a human first.
7. Do not `git commit` or `git push` unless explicitly asked.

## 5. Coding standards

- Rust edition 2024 (pinned to 1.91.0 by `rust-toolchain.toml`); formatting follows `rustfmt.toml` (2-space indent, reordered imports); clippy clean.
- **Errors**: `thiserror`; library code uses `engine::Error` (Retryable / Fatal / Config / NotFound / Conflict). No `anyhow` in library code; `anyhow` is fine in tests.
- **Async**: tokio + `async-trait`; ports must be object-safe (`Box<dyn Store>` must compile); no blocking calls inside async contexts. Heavy CPU work (brute-force vector search, chunking) may use `spawn_blocking`.
- **IDs**: UUIDv7, allocated in `engine` **before** the transaction; `Store` never generates business IDs.
- **Time**: UTC only; inject the `Clock` port; never call `Utc::now()` directly.
- **Config**: serde structs injected by the facade; `engine` reads no environment variables; thresholds / models / chunking parameters are never persisted.
- **sqlx**: use runtime `sqlx::query` bindings (avoid the `query!` macro's compile-time `DATABASE_URL` dependency); always bind parameters, never interpolate values into SQL.
- **Migrations**: append-only `NNNN_*.sql`; never edit an applied migration; update `design/memory-schema.md` before adding or changing tables/columns.
- **Logging**: `tracing`; no `println!` in library code; spans should cover bank / chunk / fact levels.
- **Public items**: write rustdoc; explain *why*, not just *what*.

## 6. Architecture guardrails (review checklist)

These are settled design decisions. Do not deviate silently:

1. **Store contract**: `commit_chunk` is the only transaction entry (tx2 writes every product and flips status atomically); `claim_chunk` claims atomically and reclaims stale `extracting` rows; `neighbors()` merges `fact_link` rows with entity adjacency derived from `fact_entity`; **work lists are always derived** (no task-queue table).
2. **The retrieval unit is the `fact` row**; chunk text is not part of the retrieval surface by default; write-time and query-time embeddings must use the same active model — **never compare across models**.
3. **Edge shapes**: `temporal` undirected edges are stored as two rows (written in pairs, pruned in pairs; the canonical-key group count is always 2); `semantic` is a directed single row (per-fact top-k budget, no reciprocity implied); `causal` is a directed single row; `entity` adjacency is never materialized — it is derived from bindings.
4. **Every table carries `bank_id` directly**; deletion and partitioning must not rely on FK cascades.
5. **Single source of truth**: do not store or materialize data that an indexed query can derive; any materialization must satisfy the `memory-schema.md` §1.3 criteria (index/constraint, named consumer, cross-backend capability).
6. **LLM output is never an idempotency key**: hashing and dedup only apply to byte-stable layers (source payload, chunk slices); fact / observation / opinion rows never carry content hashes.
7. **Do not reintroduce retired constructs** (`context` column, `text_hash`, `entity_alias`, `opinion_history`, `background_task`, `subject_entity_id`, `supersedes_id`, `embedding_model`, `directed`, etc.) unless the design docs bring them back first — re-add paths are documented in `memory-schema.md` §7.C.
8. **Terminology**: "WAL" in this project means the **transactional inbox** (not a kernel WAL). Rows are collectively called `fact` / 事实行; refer to specific kinds as world row / experience row / opinion row / observation row. Never call rows `memory`.

## 7. Testing standards

- Unit tests live beside the code (`#[cfg(test)]`); integration tests go in `crates/*/tests/`.
- Use the deterministic doubles: `FakeLlm` (scripted structured outputs), `HashEmbedder`, `IdentityReranker`, `RuleTemporalParser`, `FixedClock` (located in `engine::testing`).
- Each test gets its own in-memory SQLite database (`store::testing::memory_pool()`); no shared state.
- **No network calls in unit tests**; tests against real providers use mocks (wiremock) or are explicitly `#[ignore]`d.
- Assertions about time, randomness, and IDs must be controllable (inject `Clock`; assert semantics, not literal values).
- From P5-2 onward: run the same `Store` contract suite against both SQLite and PostgreSQL.

## 8. Key invariants (implementation self-check)

- `chunk`: `(bank_id, content_hash)` is unique; the status flip and all products commit in one transaction; a failure leaves no partial rows.
- `fact`: observation rows have `chunk_id IS NULL`; reflect-formed opinions have `chunk_id IS NULL` and `confidence NOT NULL`.
- `fact_link`: temporal edges exist only as complete pairs (no one-sided rows); semantic and causal edges are single rows.
- `entity`: `mention_count` / `first_seen` / `last_seen` stay consistent with world/experience binding counts (maintained in the same transaction).

## 9. Next step

Start with **P0-1 Workspace skeleton** (`design/phase.md`). Do not begin business pipelines until P0 is complete.
