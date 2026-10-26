-- Migration 0001: initial SQLite schema for memorybank.
--
-- Logical source of truth: design/memory-schema.md §3 (7 tables, singular names).
-- The same logical schema is implemented per backend; dialect details stay here.
--
-- Checklist against memory-schema.md §3:
--   §3.1 bank        — id PK, name UNIQUE, background, skepticism/literalism/empathy 1..=5,
--                      bias_strength 0..=1
--   §3.2 source      — id PK, bank_id, title NULL, payload_text, status{received,chunked};
--                      index (bank_id, status) for the chunking rescan
--   §3.3 chunk       — id PK, bank_id, source_id, chunk_index, text, content_hash,
--                      status{pending,extracting,done,failed}; UNIQUE(bank_id, content_hash)
--   §3.4 fact        — id PK, fact_type{world,experience,opinion,observation}, text,
--                      embedding BLOB (NULL = backfill pending), chunk_id NULL,
--                      occurred_*/mentioned_at, confidence NULL, access_count,
--                      attrs JSON; opinion-without-chunk requires confidence;
--                      observations have no chunk. BM25 corpus in fact_fts (§3.4 text).
--   §3.5 entity      — id PK, UNIQUE(bank_id, canonical_name), entity_type{...},
--                      mention_count, first_seen/last_seen, attrs JSON
--   §3.6 fact_entity — composite PK (fact_id, entity_id), bank_id, mention_text;
--                      indexes in both directions
--   §3.7 fact_link   — id PK, link_type{temporal,semantic,causal}, relation NULL,
--                      weight 0..=1; unique edge = (from, to, link_type, relation)
--                      with NULL treated as a single value (expression index)
--
-- Conventions: ids are UUIDv7 strings; timestamps are UTC epoch milliseconds;
-- enums are TEXT + CHECK; JSON is TEXT validated by json_valid(); no FK
-- constraints (every row carries bank_id and deletion is range-based, per §0.10).

CREATE TABLE bank (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  background TEXT NOT NULL,
  skepticism INTEGER NOT NULL CHECK (skepticism BETWEEN 1 AND 5),
  literalism INTEGER NOT NULL CHECK (literalism BETWEEN 1 AND 5),
  empathy INTEGER NOT NULL CHECK (empathy BETWEEN 1 AND 5),
  bias_strength REAL NOT NULL CHECK (bias_strength BETWEEN 0.0 AND 1.0),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE source (
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  title TEXT,
  payload_text TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('received', 'chunked')),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE chunk (
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
  text TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'extracting', 'done', 'failed')),
  time_start INTEGER,
  time_end INTEGER,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE (bank_id, content_hash)
);

CREATE TABLE fact (
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  fact_type TEXT NOT NULL CHECK (fact_type IN ('world', 'experience', 'opinion', 'observation')),
  text TEXT NOT NULL,
  embedding BLOB,
  chunk_id TEXT,
  occurred_start INTEGER,
  occurred_end INTEGER,
  mentioned_at INTEGER,
  confidence REAL CHECK (confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0),
  access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
  attrs TEXT CHECK (attrs IS NULL OR json_valid(attrs)),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  -- §3.4: observations never come from a chunk.
  CHECK (fact_type <> 'observation' OR chunk_id IS NULL),
  -- §3.4: reflect-produced opinions (no chunk) always carry a confidence.
  CHECK (chunk_id IS NOT NULL OR fact_type <> 'opinion' OR confidence IS NOT NULL)
);

CREATE TABLE entity (
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  canonical_name TEXT NOT NULL,
  entity_type TEXT NOT NULL
    CHECK (entity_type IN ('person', 'organization', 'location', 'product', 'concept', 'other')),
  mention_count INTEGER NOT NULL DEFAULT 0 CHECK (mention_count >= 0),
  first_seen INTEGER,
  last_seen INTEGER,
  attrs TEXT CHECK (attrs IS NULL OR json_valid(attrs)),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE (bank_id, canonical_name)
);

CREATE TABLE fact_entity (
  bank_id TEXT NOT NULL,
  fact_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  mention_text TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (fact_id, entity_id)
);

CREATE TABLE fact_link (
  id TEXT PRIMARY KEY,
  bank_id TEXT NOT NULL,
  from_fact_id TEXT NOT NULL,
  to_fact_id TEXT NOT NULL,
  link_type TEXT NOT NULL CHECK (link_type IN ('temporal', 'semantic', 'causal')),
  relation TEXT CHECK (
    relation IS NULL OR relation IN ('causes', 'caused_by', 'enables', 'prevents')
  ),
  weight REAL NOT NULL CHECK (weight BETWEEN 0.0 AND 1.0),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  -- §3.7: the relation is required for causal edges and forbidden otherwise.
  CHECK ((link_type = 'causal') = (relation IS NOT NULL))
);

-- BM25 corpus over fact.text (maintained by the store adapter, §3.4).
CREATE VIRTUAL TABLE fact_fts USING fts5(
  fact_id UNINDEXED,
  text,
  tokenize = 'trigram'
);

-- §3.7: unique edge. SQLite treats NULLs as distinct in UNIQUE constraints, so
-- non-causal edges (relation NULL) collapse onto '' to keep them deduplicated.
CREATE UNIQUE INDEX uq_fact_link_edge
  ON fact_link (from_fact_id, to_fact_id, link_type, COALESCE(relation, ''));

-- Deletion/partition and hot-path indexes (phase P0-9 checklist).
CREATE INDEX idx_fact_bank ON fact (bank_id);
CREATE INDEX idx_fact_chunk ON fact (chunk_id);
CREATE INDEX idx_fact_bank_occurred ON fact (bank_id, occurred_start);
CREATE INDEX idx_source_bank_status ON source (bank_id, status);
CREATE INDEX idx_fact_entity_bank_entity ON fact_entity (bank_id, entity_id);
CREATE INDEX idx_fact_entity_bank_fact ON fact_entity (bank_id, fact_id);
CREATE INDEX idx_fact_link_bank_from ON fact_link (bank_id, from_fact_id);
CREATE INDEX idx_fact_link_bank_to ON fact_link (bank_id, to_fact_id);
