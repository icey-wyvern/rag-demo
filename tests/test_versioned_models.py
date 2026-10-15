import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import DateTime, String, UniqueConstraint
from sqlalchemy.orm import configure_mappers

from api.db import Base
from api import versioned_models  # noqa: F401 - register canonical schema


def unique_columns(table):
    return {
        tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }


class VersionedModelTests(unittest.TestCase):
    def test_source_identity_and_uniqueness(self):
        table = Base.metadata.tables["sources"]
        self.assertEqual(table.primary_key.columns.keys(), ["id"])
        self.assertIn(("source_key",), unique_columns(table))
        self.assertFalse(table.c.canonical_url.nullable)
        self.assertFalse(table.c.title.nullable)

    def test_snapshot_source_hash_and_same_source_predecessor(self):
        table = Base.metadata.tables["source_snapshots"]
        self.assertIn(("source_id", "content_sha256"), unique_columns(table))
        self.assertIn(("source_id", "id"), unique_columns(table))
        self.assertFalse(table.c.source_id.nullable)
        self.assertEqual(table.c.content_sha256.type.length, 64)

        foreign_keys = {constraint.name: constraint for constraint in table.foreign_key_constraints}
        self.assertEqual(foreign_keys["fk_source_snapshots_source"].referred_table.name, "sources")
        predecessor = foreign_keys["fk_source_snapshots_predecessor_same_source"]
        self.assertEqual(set(predecessor.column_keys), {"source_id", "predecessor_snapshot_id"})
        self.assertEqual(predecessor.referred_table.name, "source_snapshots")
        self.assertEqual(predecessor.ondelete, "RESTRICT")
        configure_mappers()
        self.assertIsNotNone(versioned_models.SourceSnapshot.predecessor.property)

    def test_passages_are_bound_to_one_snapshot(self):
        table = Base.metadata.tables["passages"]
        self.assertIn(("snapshot_id", "ordinal"), unique_columns(table))
        self.assertFalse(table.c.snapshot_id.nullable)
        self.assertTrue(table.c.page.nullable)
        self.assertFalse(table.c.text.nullable)
        self.assertIsInstance(table.c.text_sha256.type, String)
        self.assertEqual(table.c.text_sha256.type.length, 64)
        foreign_key = next(iter(table.foreign_key_constraints))
        self.assertEqual(foreign_key.referred_table.name, "source_snapshots")
        self.assertEqual(foreign_key.ondelete, "RESTRICT")

    def test_timestamps_are_timezone_aware_and_metadata_is_canonical(self):
        timestamp_columns = {
            "sources": ["created_at"],
            "source_snapshots": [
                "observed_at", "source_published_at", "source_updated_at", "effective_from", "created_at"
            ],
            "passages": ["created_at"],
        }
        for table_name, columns in timestamp_columns.items():
            table = Base.metadata.tables[table_name]
            for column_name in columns:
                with self.subTest(table=table_name, column=column_name):
                    column = table.c[column_name]
                    self.assertIsInstance(column.type, DateTime)
                    self.assertTrue(column.type.timezone)
        self.assertEqual(set(Base.metadata.tables),
                         {"sources", "source_snapshots", "passages", "passage_embeddings"})

    def test_alembic_upgrade_renders_legacy_cleanup(self):
        repository = Path(__file__).resolve().parents[1]
        config = Config(str(repository / "alembic.ini"))
        config.set_main_option("script_location", str(repository / "migrations"))
        output = io.StringIO()
        with redirect_stdout(output):
            command.upgrade(config, "head", sql=True)
        sql = output.getvalue()
        for table in ("sources", "source_snapshots", "passages"):
            self.assertIn(f"CREATE TABLE {table}", sql)
        for legacy_table in ("documents", "chunks", "corpus_state"):
            self.assertIn(f"DROP TABLE IF EXISTS {legacy_table}", sql)


if __name__ == "__main__":
    unittest.main()
