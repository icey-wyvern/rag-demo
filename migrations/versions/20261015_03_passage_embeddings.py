"""Add model-specific embeddings for immutable passages.

Revision ID: 20261015_03
Revises: 20261015_02
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


revision = "20261015_03"
down_revision = "20261015_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "passage_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("passage_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_passage_embeddings"),
        sa.ForeignKeyConstraint(["passage_id"], ["passages.id"], name="fk_passage_embeddings_passage", ondelete="CASCADE"),
        sa.UniqueConstraint("passage_id", "provider", "model", name="uq_passage_embeddings_identity"),
        sa.CheckConstraint("dimensions = 1536", name="ck_passage_embeddings_dimensions_1536"),
    )
    op.create_index("ix_passage_embeddings_provider_model", "passage_embeddings", ["provider", "model"])


def downgrade() -> None:
    op.drop_index("ix_passage_embeddings_provider_model", table_name="passage_embeddings")
    op.drop_table("passage_embeddings")
