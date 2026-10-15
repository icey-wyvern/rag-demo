"""Index versioned passages for PostgreSQL English full-text search.

Revision ID: 20261015_02
Revises: 20261015_01
"""

from alembic import op


revision = "20261015_02"
down_revision = "20261015_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_passages_text_english_fts "
        "ON passages USING gin (to_tsvector('english', text))"
    )


def downgrade() -> None:
    op.drop_index("ix_passages_text_english_fts", table_name="passages")
