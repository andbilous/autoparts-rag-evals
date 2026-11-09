CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    doc_key       TEXT NOT NULL,              -- стабільний ключ документа (не змінюється між версіями)
    file_name     TEXT NOT NULL,
    title         TEXT NOT NULL,
    manufacturer  TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    articles      TEXT[] NOT NULL DEFAULT '{}', -- нормалізовані артикули товарів
    content_hash  TEXT NOT NULL,
    version       INT NOT NULL DEFAULT 1,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    storage_path  TEXT NOT NULL,
    page_count    INT NOT NULL,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS documents_one_active ON documents (doc_key) WHERE is_active;
CREATE INDEX IF NOT EXISTS documents_articles ON documents USING GIN (articles);

CREATE TABLE IF NOT EXISTS chunks (
    id             BIGSERIAL PRIMARY KEY,
    document_id    BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index    INT NOT NULL,
    page_from      INT NOT NULL,
    page_to        INT NOT NULL,
    section_title  TEXT,
    chunk_type     TEXT NOT NULL CHECK (chunk_type IN ('text', 'table', 'figure_caption')),
    content        TEXT NOT NULL,
    content_lemmas TEXT NOT NULL,
    tsv            TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content_lemmas)) STORED,
    articles_norm  TEXT[] NOT NULL DEFAULT '{}',
    embedding      VECTOR(1024) NOT NULL,
    content_hash   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_articles ON chunks USING GIN (articles_norm);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks (document_id);

CREATE TABLE IF NOT EXISTS ocr_review_queue (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INT NOT NULL,
    confidence  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_session ON messages (session_id, id);

CREATE TABLE IF NOT EXISTS traces (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    question        TEXT NOT NULL,
    intent          JSONB,
    rewritten_query TEXT,
    query_articles  TEXT[],
    candidates      JSONB,
    outcome         TEXT,          -- answer | refuse_no_info | refuse_out_of_scope | refuse_abuse | retracted | error
    answer          TEXT,
    citations       JSONB,
    verification    JSONB,
    usage           JSONB,
    cost_usd        NUMERIC(12, 6),
    latency_ms      JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id         BIGSERIAL PRIMARY KEY,
    trace_id   BIGINT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
    useful     BOOLEAN NOT NULL,
    comment    TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
