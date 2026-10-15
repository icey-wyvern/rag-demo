"""Offline tests for model-specific versioned passage indexing."""

import copy
import hashlib
import math
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, UniqueConstraint
from sqlalchemy.dialects import postgresql

from api.db import Base
from api.passage_embedding_indexer import EmbeddingIndexError, PassageEmbeddingIndexer
from api.passage_embedding_provider import EMBEDDING_DIMENSIONS, OpenAIPassageEmbeddingProvider
from api.passage_embedding_store import PassageEmbeddingStore
from api.versioned_models import PassageEmbedding


def passage(number, text, snapshot=1):
    return SimpleNamespace(id=UUID(int=number), snapshot_id=UUID(int=snapshot), text=text,
                           text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


class MemorySessions:
    """Copy-on-write transaction fake; it does not emulate pgvector or row locks."""

    def __init__(self, passages):
        self.state = SimpleNamespace(passages=list(passages), embeddings=[])
        self.commits = 0
        self.rollbacks = 0
        self.fail_commit = False
        self.fail_commit_on = None

    @contextmanager
    def begin(self):
        working = copy.deepcopy(self.state)
        try:
            yield working
            if self.fail_commit or self.fail_commit_on == self.commits + 1:
                raise RuntimeError("commit failed")
        except Exception:
            self.rollbacks += 1
            raise
        else:
            self.state = working
            self.commits += 1


class MemoryStore:
    fail_persistence = False

    def __init__(self, state):
        self.state = state

    def count_passages(self):
        return len(self.state.passages)

    def count_indexed(self, provider, model):
        return sum(row.provider == provider and row.model == model for row in self.state.embeddings)

    def missing_batch(self, provider, model, limit):
        indexed = {row.passage_id for row in self.state.embeddings if row.provider == provider and row.model == model}
        return sorted((p for p in self.state.passages if p.id not in indexed), key=lambda p: p.id)[:limit]

    def add_batch(self, provider, model, vectors):
        for passage_id, vector in vectors:
            self.state.embeddings.append(SimpleNamespace(
                passage_id=passage_id, provider=provider, model=model,
                dimensions=len(vector), embedding=vector,
            ))
            if self.fail_persistence:
                raise RuntimeError("persistence failed")


class DeterministicFakeProvider:
    """Test-only stable bytes-to-vector mapping; not a semantic embedding."""

    provider_id = "fixture"
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, model_id="fixture-v1"):
        self.model_id = model_id
        self.calls = []
        self.override = None
        self.failure = None

    def embed_texts(self, texts):
        self.calls.append(tuple(texts))
        if self.failure:
            raise self.failure
        if self.override is not None:
            return self.override
        return [[hashlib.sha256(text.encode("utf-8")).digest()[i % 32] / 255.0
                 for i in range(self.dimensions)] for text in texts]


class IndexerTests(unittest.TestCase):
    def setUp(self):
        self.sessions = MemorySessions([passage(1, "First guidance"), passage(2, "Second guidance")])
        self.provider = DeterministicFakeProvider()
        self.indexer = PassageEmbeddingIndexer(
            self.sessions, self.provider, batch_size=2, store_factory=MemoryStore,
        )

    def test_first_indexing_and_exact_rerun(self):
        original_passages = copy.deepcopy(self.sessions.state.passages)
        first = self.indexer.index_missing()
        self.assertEqual(self.provider.calls, [("First guidance", "Second guidance")])
        self.assertEqual((first.total_passages, first.already_indexed, first.newly_indexed, first.remaining, first.batches),
                         (2, 0, 2, 0, 1))
        self.assertEqual(len(self.sessions.state.embeddings), 2)
        self.assertEqual([row.passage_id for row in self.sessions.state.embeddings], [UUID(int=1), UUID(int=2)])
        self.assertTrue(all(row.provider == "fixture" and row.model == "fixture-v1" and row.dimensions == 1536
                            for row in self.sessions.state.embeddings))
        self.assertEqual(self.sessions.state.passages, original_passages)
        repeat = self.indexer.index_missing()
        self.assertEqual((repeat.already_indexed, repeat.newly_indexed, repeat.batches), (2, 0, 0))
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(len(self.sessions.state.embeddings), 2)

    def test_later_snapshot_only_indexes_new_passages(self):
        self.indexer.index_missing()
        prior = copy.deepcopy(self.sessions.state.embeddings)
        self.sessions.state.passages.append(passage(3, "Updated guidance", snapshot=2))
        result = self.indexer.index_missing()
        self.assertEqual((result.already_indexed, result.newly_indexed), (2, 1))
        self.assertEqual(self.provider.calls[-1], ("Updated guidance",))
        self.assertEqual(self.sessions.state.embeddings[:2], prior)
        self.assertEqual(self.sessions.state.embeddings[2].passage_id, UUID(int=3))

    def test_different_model_identity_coexists(self):
        self.indexer.index_missing()
        prior = copy.deepcopy(self.sessions.state.embeddings)
        other = DeterministicFakeProvider("fixture-v2")
        result = PassageEmbeddingIndexer(self.sessions, other, store_factory=MemoryStore).index_missing()
        self.assertEqual(result.newly_indexed, 2)
        self.assertEqual(self.sessions.state.embeddings[:2], prior)
        self.assertEqual({row.model for row in self.sessions.state.embeddings}, {"fixture-v1", "fixture-v2"})

    def test_batches_commit_independently(self):
        self.sessions.state.passages.append(passage(3, "Third guidance", snapshot=2))
        indexer = PassageEmbeddingIndexer(self.sessions, self.provider, batch_size=2, store_factory=MemoryStore)
        result = indexer.index_missing()
        self.assertEqual((result.newly_indexed, result.batches), (3, 2))
        self.assertEqual([len(call) for call in self.provider.calls], [2, 1])

    def test_later_batch_failure_preserves_earlier_committed_batch(self):
        self.sessions.state.passages.append(passage(3, "Third guidance", snapshot=2))
        class FailsSecondBatch(DeterministicFakeProvider):
            def embed_texts(self, texts):
                if self.calls:
                    self.calls.append(tuple(texts))
                    raise RuntimeError("second batch failed")
                return super().embed_texts(texts)
        provider = FailsSecondBatch()
        with self.assertRaisesRegex(EmbeddingIndexError, "provider failed"):
            PassageEmbeddingIndexer(self.sessions, provider, batch_size=2, store_factory=MemoryStore).index_missing()
        self.assertEqual([row.passage_id for row in self.sessions.state.embeddings], [UUID(int=1), UUID(int=2)])
        self.assertEqual(self.sessions.rollbacks, 1)

    def test_wrong_dimensions_count_or_numbers_fail_without_persistence(self):
        cases = [
            ([[0.1] * 3, [0.2] * 1536], "dimensions"),
            ([[0.1] * 1536], "vectors for 2"),
            ([[0.1] * 1536] * 3, "vectors for 2"),
            ([[math.nan] + [0.1] * 1535, [0.2] * 1536], "non-finite"),
            ([[math.inf] + [0.1] * 1535, [0.2] * 1536], "non-finite"),
            ([["bad"] + [0.1] * 1535, [0.2] * 1536], "non-numeric"),
        ]
        for vectors, message in cases:
            with self.subTest(message=message):
                sessions = MemorySessions([passage(1, "One"), passage(2, "Two")])
                provider = DeterministicFakeProvider()
                provider.override = vectors
                with self.assertRaisesRegex(EmbeddingIndexError, message):
                    PassageEmbeddingIndexer(sessions, provider, store_factory=MemoryStore).index_missing()
                self.assertEqual(sessions.state.embeddings, [])
                self.assertEqual(sessions.rollbacks, 1)

    def test_provider_persistence_and_commit_failures_rollback_batch(self):
        self.provider.failure = RuntimeError("provider offline")
        with self.assertRaisesRegex(EmbeddingIndexError, "provider failed"):
            self.indexer.index_missing()
        self.assertEqual(self.sessions.state.embeddings, [])
        self.assertEqual(self.sessions.rollbacks, 1)
        self.provider.failure = None
        class FailingStore(MemoryStore):
            fail_persistence = True
        with self.assertRaisesRegex(RuntimeError, "persistence failed"):
            PassageEmbeddingIndexer(self.sessions, self.provider, store_factory=FailingStore).index_missing()
        self.assertEqual(self.sessions.state.embeddings, [])
        self.assertEqual(self.sessions.rollbacks, 2)
        self.sessions.fail_commit_on = self.sessions.commits + 2
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            self.indexer.index_missing()
        self.assertEqual(self.sessions.state.embeddings, [])
        self.assertEqual(self.sessions.rollbacks, 3)

    def test_invalid_provider_and_batch_configuration(self):
        self.provider.dimensions = 3
        with self.assertRaisesRegex(ValueError, "1536"):
            PassageEmbeddingIndexer(self.sessions, self.provider)
        self.provider.dimensions = 1536
        for size in (0, 101, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "batch_size"):
                PassageEmbeddingIndexer(self.sessions, self.provider, batch_size=size)


class ModelAndProviderTests(unittest.TestCase):
    def test_schema_is_separate_model_specific_and_bounded_to_1536(self):
        table = Base.metadata.tables["passage_embeddings"]
        self.assertNotIn("embedding", Base.metadata.tables["passages"].c)
        self.assertIsInstance(table.c.embedding.type, Vector)
        self.assertEqual(table.c.embedding.type.dim, 1536)
        self.assertEqual(table.c.dimensions.nullable, False)
        self.assertIsInstance(table.c.created_at.type, DateTime)
        self.assertTrue(table.c.created_at.type.timezone)
        self.assertTrue(any(isinstance(c, UniqueConstraint) and c.name == "uq_passage_embeddings_identity"
                            and {col.name for col in c.columns} == {"passage_id", "provider", "model"}
                            for c in table.constraints))
        self.assertTrue(any(isinstance(c, CheckConstraint) and c.name == "ck_passage_embeddings_dimensions_1536"
                            for c in table.constraints))
        foreign_key = next(iter(table.c.passage_id.foreign_keys))
        self.assertEqual(foreign_key.ondelete, "CASCADE")
        self.assertEqual(foreign_key.column.table.name, "passages")
        self.assertIn("passage_embeddings", Base.metadata.tables)
        self.assertTrue({"documents", "chunks", "sources", "source_snapshots", "passages"} <= set(Base.metadata.tables))

    def test_sql_missing_query_uses_identity_and_skip_locked(self):
        session = Mock()
        session.execute.return_value.all.return_value = []
        PassageEmbeddingStore(session).missing_batch("openai", "text-embedding-3-small", 32)
        statement = session.execute.call_args.args[0]
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertIn("LEFT OUTER JOIN passage_embeddings", sql)
        self.assertIn("passage_embeddings.provider", sql)
        self.assertIn("passage_embeddings.model", sql)
        self.assertIn("passage_embeddings.id IS NULL", sql)
        self.assertIn("ORDER BY passages.id", sql)
        self.assertIn("FOR UPDATE OF passages SKIP LOCKED", sql)

    def test_sqlalchemy_store_creates_model_specific_rows(self):
        session = Mock()
        vector = [0.25] * 1536
        PassageEmbeddingStore(session).add_batch(
            "openai", "text-embedding-3-small", [(UUID(int=7), vector)],
        )
        rows = session.add_all.call_args.args[0]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIsInstance(row, PassageEmbedding)
        self.assertEqual((row.passage_id, row.provider, row.model, row.dimensions),
                         (UUID(int=7), "openai", "text-embedding-3-small", 1536))
        self.assertEqual(row.embedding, vector)
        session.flush.assert_called_once()

    def test_openai_adapter_uses_configuration_and_preserves_order_without_network(self):
        provider = OpenAIPassageEmbeddingProvider(model="text-embedding-3-small", api_key="test-only-key", timeout_seconds=7)
        client = Mock()
        client.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
        with patch("langchain_openai.OpenAIEmbeddings", return_value=client) as factory:
            result = provider.embed_texts(["first", "second"])
        factory.assert_called_once_with(
            model="text-embedding-3-small", dimensions=1536,
            openai_api_key="test-only-key", timeout=7, max_retries=2,
        )
        client.embed_documents.assert_called_once_with(["first", "second"])
        self.assertEqual(result[0][0], 0.1)
        self.assertEqual(result[1][0], 0.2)


if __name__ == "__main__":
    unittest.main()
