"""Add versioned corpus tables alongside the legacy corpus.

Revision ID: 20261015_01
Revises:
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20261015_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_key", sa.Text(), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_sources"),
        sa.UniqueConstraint("source_key", name="uq_sources_source_key"),
    )
    op.create_table(
        "source_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("retrieved_url", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("predecessor_snapshot_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_source_snapshots"),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], name="fk_source_snapshots_source", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["source_id", "predecessor_snapshot_id"],
            ["source_snapshots.source_id", "source_snapshots.id"],
            name="fk_source_snapshots_predecessor_same_source",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("source_id", "content_sha256", name="uq_source_snapshots_source_hash"),
        sa.UniqueConstraint("source_id", "id", name="uq_source_snapshots_source_id_id"),
    )
    op.create_index(
        "ix_source_snapshots_source_observed_at",
        "source_snapshots",
        ["source_id", "observed_at"],
    )
    op.create_table(
        "passages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("snapshot_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_passages"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["source_snapshots.id"], name="fk_passages_snapshot", ondelete="RESTRICT"),
        sa.UniqueConstraint("snapshot_id", "ordinal", name="uq_passages_snapshot_ordinal"),
    )


def downgrade() -> None:
    op.drop_table("passages")
    op.drop_index("ix_source_snapshots_source_observed_at", table_name="source_snapshots")
    op.drop_table("source_snapshots")
    op.drop_table("sources")
