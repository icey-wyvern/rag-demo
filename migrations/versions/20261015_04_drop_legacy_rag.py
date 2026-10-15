"""Remove obsolete legacy RAG tables; Alembic owns the application schema.

Revision ID: 20261015_04
Revises: 20261015_03

Upgrading permanently deletes data in corpus_state, documents, and chunks if
those old tables exist. Versioned sources, snapshots, passages, and embeddings
are untouched. Downgrade recreates the old schema, not its deleted rows.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


revision = "20261015_04"
down_revision = "20261015_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # These tables came from PostgreSQL init SQL, not earlier Alembic revisions.
    # A clean Alembic-only database has none of them; an existing installation
    # may require running this one-time cleanup as their owning database role.
    for index in (
        "idx_chunks_embedding_hnsw", "idx_chunks_doc_page",
        "idx_documents_base_path", "idx_documents_status",
    ):
        op.execute(f"DROP INDEX IF EXISTS {index}")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS corpus_state")


def downgrade() -> None:
    op.create_table(
        "corpus_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("ingest_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("documents_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("documents_ready", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("base_path", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("source_sha256", sa.Text()),
        sa.Column("ingest_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("base_path", name="documents_base_path_key"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("doc_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ord", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("doc_id", "ord", name="chunks_doc_id_ord_key"),
    )
    op.create_index("idx_documents_base_path", "documents", ["base_path"])
    op.create_index("idx_documents_status", "documents", ["status"])
    op.create_index("idx_chunks_doc_page", "chunks", ["doc_id", "page"])
    op.execute("CREATE INDEX idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_l2_ops)")
