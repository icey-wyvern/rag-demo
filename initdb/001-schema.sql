-- Schema for the RAG demo.
\connect appdb

-- Extensions (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS corpus_state (
	id              int PRIMARY KEY,
	status          text NOT NULL DEFAULT 'pending',
	ingest_version  int NOT NULL DEFAULT 1,
	documents_total int NOT NULL DEFAULT 0,
	documents_ready int NOT NULL DEFAULT 0,
	last_error      text,
	updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
	id         uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
	title      text NOT NULL,
	base_path  text NOT NULL UNIQUE,
	url        text NOT NULL,
	source_sha256 text,
	ingest_version int NOT NULL DEFAULT 1,
	chunk_count int NOT NULL DEFAULT 0,
	status     text NOT NULL DEFAULT 'pending',
	created_at timestamptz NOT NULL DEFAULT now(),
	updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
	id        uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
	doc_id    uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
	ord       int NOT NULL,
	page      int NOT NULL,
	content   text NOT NULL,
	embedding vector(1536) NOT NULL,
	created_at timestamptz NOT NULL DEFAULT now(),
	UNIQUE (doc_id, ord)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_documents_base_path ON documents (base_path);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks (doc_id, page);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
	ON chunks USING hnsw (embedding vector_l2_ops);
