-- Schema for RAG demo (documents + chunks with pgvector embeddings)
-- Ensures all objects are created in the intended database.
\connect appdb

-- Extensions (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents: one row per GOV.UK page
CREATE TABLE IF NOT EXISTS documents (
	id         uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
	title      text NOT NULL,
	base_path  text NOT NULL,
	url        text NOT NULL,
	created_at timestamptz NOT NULL DEFAULT now()
);

-- Chunks: denormalised text segments + embedding for vector search
CREATE TABLE IF NOT EXISTS chunks (
	id        uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
	doc_id    uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
	ord       int  NOT NULL,                         -- original order within a doc
	content   text NOT NULL,                         -- cleaned paragraph(s)
	embedding vector(1536)                           -- OpenAI text-embedding-3-* size
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_chunks_doc_ord ON chunks (doc_id, ord);
CREATE INDEX IF NOT EXISTS idx_documents_base_path ON documents (base_path);

-- Vector index (choose ONE; HNSW recommended)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_l2_ops);

-- Alternative (IVFFlat) — comment out HNSW above if you use this:
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
-- ON chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
