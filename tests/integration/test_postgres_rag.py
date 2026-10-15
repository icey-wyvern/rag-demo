"""Real PostgreSQL/pgvector path; opt in with a guarded disposable test URL.

The test database must be empty. This suite never drops a database or tables.
"""

import json
import os
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from api.content_hash import sha256_text
from api.evidence import Evidence
from api.evidence_answer import EvidenceFirstAnswerService
from api.hybrid_rrf import candidate_depth
from api.main import app, get_query_pipeline, get_versioned_status
from api.passage_embedding_indexer import PassageEmbeddingIndexer
from api.query_service import QueryPipeline
from api.versioned_ingest import ObservedSourceVersion, VersionedIngestionService
from api.versioned_models import Passage, PassageEmbedding, Source, SourceSnapshot
from api.versioned_parser import ParsedPassage
from api.versioned_readiness import versioned_corpus_state
from api.versioned_retrieval_store import VersionedRetrievalStore
from api.versioned_retriever import VersionedRetriever


OLD = datetime(2026, 10, 15, 10, tzinfo=timezone.utc)
MIDDLE = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
NEW = datetime(2026, 10, 16, 10, tzinfo=timezone.utc)
BEFORE = datetime(2026, 10, 15, 9, tzinfo=timezone.utc)
MODEL = "integration-fixture"


def guarded_url(raw: str):
    """Refuse all normal development URLs before connecting or migrating."""
    url = make_url(raw)
    if (url.drivername != "postgresql+psycopg" or url.database != "rag_integration_test"):
        raise ValueError("Integration tests require the dedicated PostgreSQL test database")
    if (url.username != "rag_test" or url.password != "rag_test_password"
            or url.host not in ("localhost", "127.0.0.1")):
        raise ValueError("Integration tests require the local test-only role and host")
    if url.port not in (5432, 55439):
        raise ValueError("Integration tests require the declared test port")
    return url


def vector(x: float, y: float = 0.0) -> list[float]:
    return [x, y] + [0.0] * 1534


class FakeEmbeddings:
    provider_id = "openai"  # Matches production readiness filtering; no OpenAI client is used.
    model_id = MODEL
    dimensions = 1536

    def embed_texts(self, texts):
        result = []
        for passage in texts:
            if passage.startswith("Tidal"):
                result.append(vector(0.0, 1.0))
            elif passage.startswith("Harborplan partner"):
                result.append(vector(0.6, 0.8))
            else:
                result.append(vector(1.0))
        return result


class FakeAnswers:
    def __init__(self):
        self.calls = 0

    def generate(self, question, formatted_evidence, instructions):
        self.calls += 1
        first = json.loads(formatted_evidence)[0]
        quote = first["passage_text"].split()[0]
        return {
            "status": "answered", "abstention_reason": None,
            "claims": [{"text": f"The observed guidance mentions {quote}.",
                        "citations": [{"evidence_id": first["evidence_id"],
                                       "supporting_quote": quote}]}],
        }


@unittest.skipUnless(os.getenv("RAG_TEST_DATABASE_URL"), "Set RAG_TEST_DATABASE_URL for disposable PostgreSQL integration tests")
class PostgreSQLRagIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        raw = os.environ["RAG_TEST_DATABASE_URL"]
        guarded_url(raw)  # Must precede any connection or migration.
        cls.engine = create_engine(raw, future=True)
        cls.sessions = sessionmaker(cls.engine, expire_on_commit=False, future=True)
        try:
            with cls.engine.connect() as connection:
                identity = connection.execute(text("SELECT current_database(), current_user")).one()
                if tuple(identity) != ("rag_integration_test", "rag_test"):
                    raise RuntimeError("Connected database identity is not the isolated test database")
                if inspect(connection).get_table_names():
                    raise RuntimeError("Integration database is not empty; refusing to alter existing tables")
            # This only creates an extension in the guarded, empty test database.
            with cls.engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            import api.config as config
            with patch.object(config, "_settings", replace(config.get_settings(), DATABASE_URL=raw)):
                alembic = Config("alembic.ini")
                command.upgrade(alembic, "head")
                with cls.engine.connect() as connection:
                    cls.initial_head_tables = set(inspect(connection).get_table_names())
                # Exercise the full reversible chain before fixture data exists.
                command.downgrade(alembic, "base")
                with cls.engine.connect() as connection:
                    cls.downgraded_legacy_tables = set(inspect(connection).get_table_names())
                command.upgrade(alembic, "head")

            with cls.sessions() as session:
                cls.empty_readiness = versioned_corpus_state(session, MODEL, 1, provider_configured=True)

            parser = lambda content: [ParsedPassage(page=1, text=value) for value in content.decode().split("|")]
            ingest = VersionedIngestionService(cls.sessions, parser=parser)
            def observation(key, title, content, observed):
                return ObservedSourceVersion(
                    source_key=key, canonical_url=f"https://example.invalid/{key}", title=title,
                    retrieved_url=f"https://example.invalid/{key}", content=content.encode(),
                    observed_at=observed, source_updated_at=observed,
                )

            cls.old = ingest.ingest(observation("a", "Harbor guidance", "Heritageharbor guidance requires alpha review.", OLD))
            cls.repeat = ingest.ingest(observation("a", "Harbor guidance", "Heritageharbor guidance requires alpha review.", OLD))
            cls.other = ingest.ingest(observation("b", "Partner guidance", "Harborplan partner guidance teams publish updates.", MIDDLE))
            with cls.sessions() as session:
                cls.before_embeddings = versioned_corpus_state(session, MODEL, 1, provider_configured=True)
            cls.indexer = PassageEmbeddingIndexer(cls.sessions, FakeEmbeddings(), batch_size=2)
            cls.first_index = cls.indexer.index_missing()
            with cls.sessions() as session:
                cls.initial_embedding_ids = set(session.scalars(select(PassageEmbedding.id)).all())
            cls.new = ingest.ingest(observation(
                "a", "Harbor guidance", "Harborplan guidance requires beta review.|Tidal resilience teams coordinate drills.", NEW,
            ))
            cls.second_index = cls.indexer.index_missing()
            cls.repeat_index = cls.indexer.index_missing()
            cls.provider = FakeEmbeddings()
            cls.retriever = VersionedRetriever(cls.sessions, embedding_provider=cls.provider)
        except Exception:
            cls.engine.dispose()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        super().tearDownClass()

    def test_01_migration_schema_indexes_constraints_and_dimension(self):
        with self.engine.connect() as connection:
            self.assertEqual(connection.scalar(text("SELECT extname FROM pg_extension WHERE extname='vector'")), "vector")
            self.assertEqual(connection.scalar(text("SELECT version_num FROM alembic_version")), "20261015_04")
            tables = set(inspect(connection).get_table_names())
            self.assertTrue({"sources", "source_snapshots", "passages", "passage_embeddings"} <= tables)
            self.assertFalse({"documents", "chunks", "corpus_state"} & tables)
            self.assertTrue({"sources", "source_snapshots", "passages", "passage_embeddings"}
                            <= self.initial_head_tables)
            self.assertFalse({"documents", "chunks", "corpus_state"} & self.initial_head_tables)
            self.assertTrue({"documents", "chunks", "corpus_state"} <= self.downgraded_legacy_tables)
            indexes = {row["name"]: row for row in inspect(connection).get_indexes("passages")}
            self.assertIn("ix_passages_text_english_fts", indexes)
            self.assertEqual(indexes["ix_passages_text_english_fts"]["dialect_options"]["postgresql_using"], "gin")
            constraints = {row["name"] for table in ("source_snapshots", "passages", "passage_embeddings")
                           for row in inspect(connection).get_unique_constraints(table)}
            self.assertTrue({"uq_source_snapshots_source_hash", "uq_passages_snapshot_ordinal",
                             "uq_passage_embeddings_identity"} <= constraints)
            self.assertIn("ix_source_snapshots_source_observed_at",
                          {row["name"] for row in inspect(connection).get_indexes("source_snapshots")})
            self.assertIn("ix_passage_embeddings_provider_model",
                          {row["name"] for row in inspect(connection).get_indexes("passage_embeddings")})
            self.assertIn("ck_passage_embeddings_dimensions_1536",
                          {row["name"] for row in inspect(connection).get_check_constraints("passage_embeddings")})
            dimension = connection.scalar(text("""
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = 'passage_embeddings' AND a.attname = 'embedding'
            """))
            self.assertEqual(dimension, "vector(1536)")

    def test_02_real_ingestion_and_predecessor(self):
        self.assertEqual((self.old.status, self.repeat.status, self.new.status, self.other.status),
                         ("new_source", "unchanged", "new_snapshot", "new_source"))
        self.assertEqual(self.repeat.snapshot_id, self.old.snapshot_id)
        with self.sessions() as session:
            self.assertEqual(session.scalar(select(func.count(Source.id))), 2)
            self.assertEqual(session.scalar(select(func.count(SourceSnapshot.id))), 3)
            self.assertEqual(session.scalar(select(func.count(Passage.id))), 4)
            changed = session.get(SourceSnapshot, self.new.snapshot_id)
            self.assertEqual(changed.predecessor_snapshot_id, self.old.snapshot_id)
            self.assertEqual(changed.source_id, self.old.source_id)
            self.assertNotEqual(changed.source_id, self.other.source_id)
            self.assertEqual(session.scalar(select(func.count(Passage.id)).where(Passage.snapshot_id == self.old.snapshot_id)), 1)

    def test_03_embedding_persistence_idempotency_and_uniqueness(self):
        self.assertEqual((self.first_index.newly_indexed, self.second_index.newly_indexed,
                          self.repeat_index.newly_indexed), (2, 2, 0))
        with self.sessions() as session:
            rows = session.scalars(select(PassageEmbedding)).all()
            self.assertEqual(len(rows), 4)
            self.assertTrue(self.initial_embedding_ids <= {row.id for row in rows})
            self.assertTrue(all((row.provider, row.model, row.dimensions) == ("openai", MODEL, 1536) for row in rows))
            self.assertTrue(all(len(row.embedding) == 1536 for row in rows))
            passage_id = rows[0].passage_id
        with self.assertRaises(IntegrityError):
            with self.sessions.begin() as session:
                session.add(PassageEmbedding(passage_id=passage_id, provider="openai", model=MODEL,
                                             dimensions=1536, embedding=vector(1.0)))

    def test_04_real_readiness_transitions(self):
        self.assertFalse(self.empty_readiness["versioned"]["lexical_ready"])
        self.assertFalse(self.empty_readiness["versioned"]["semantic_ready"])
        self.assertTrue(self.before_embeddings["versioned"]["lexical_ready"])
        self.assertFalse(self.before_embeddings["versioned"]["semantic_ready"])
        with self.sessions() as session:
            ready = versioned_corpus_state(session, MODEL, 1, provider_configured=True)
        self.assertTrue(ready["is_ready"])
        self.assertEqual(ready["versioned"]["current_passages"], 3)
        self.assertEqual(ready["versioned"]["embedded_passages"], 3)

    def test_05_postgresql_lexical_and_temporal_eligibility(self):
        current = self.retriever.search("harborplan", mode="lexical")
        self.assertEqual([hit.source_key for hit in current], ["a", "b"])
        self.assertEqual(current[0].snapshot_id, self.new.snapshot_id)
        self.assertEqual([hit.passage_id for hit in current],
                         [hit.passage_id for hit in self.retriever.search("harborplan", mode="lexical")])
        self.assertTrue(all(hit.retrieval_method == "postgresql_fts_english" for hit in current))
        self.assertEqual(self.retriever.search("heritageharbor", mode="lexical"), [])
        old = self.retriever.search("heritageharbor", mode="lexical", as_of=MIDDLE)
        self.assertEqual([hit.snapshot_id for hit in old], [self.old.snapshot_id])
        self.assertEqual(self.retriever.search("heritageharbor", mode="lexical", as_of=BEFORE), [])
        self.assertEqual(self.retriever.search("harborplan", mode="lexical", as_of=OLD), [])
        both = self.retriever.search("guidance", mode="lexical", as_of=MIDDLE)
        self.assertEqual({hit.snapshot_id for hit in both}, {self.old.snapshot_id, self.other.snapshot_id})

    def test_06_pgvector_cosine_provider_filter_and_temporal_eligibility(self):
        current = self.retriever.search("harborplan", mode="semantic", k=3)
        self.assertEqual(current[0].snapshot_id, self.new.snapshot_id)
        self.assertEqual(current[0].passage_ordinal, 0)
        self.assertEqual(current[0].score, 0.0)
        self.assertLess(current[1].score, current[2].score)
        self.assertTrue(all(hit.retrieval_method == "pgvector_cosine_exact" for hit in current))
        old = self.retriever.search("harborplan", mode="semantic", as_of=MIDDLE)
        self.assertEqual(old[0].snapshot_id, self.old.snapshot_id)
        self.assertNotIn(self.new.snapshot_id, {hit.snapshot_id for hit in old})
        self.assertEqual({hit.snapshot_id for hit in self.retriever.search("harborplan", mode="semantic", as_of=OLD)},
                         {self.old.snapshot_id})
        self.assertEqual(self.retriever.search("harborplan", mode="semantic", as_of=BEFORE), [])
        with self.sessions.begin() as session:
            passage = session.scalar(select(Passage).where(Passage.snapshot_id == self.other.snapshot_id))
            session.add_all([
                PassageEmbedding(passage_id=passage.id, provider="openai", model="wrong-model",
                                 dimensions=1536, embedding=vector(1.0)),
                PassageEmbedding(passage_id=passage.id, provider="other", model=MODEL,
                                 dimensions=1536, embedding=vector(1.0)),
            ])
        after = self.retriever.search("harborplan", mode="semantic", k=3)
        self.assertEqual([hit.passage_id for hit in after], [hit.passage_id for hit in current])

    def test_07_real_hybrid_rrf_and_provenance(self):
        hits = self.retriever.search("harborplan", mode="hybrid", k=3)
        self.assertEqual(hits[0].snapshot_id, self.new.snapshot_id)
        self.assertEqual((hits[0].lexical_rank, hits[0].semantic_rank), (1, 1))
        self.assertAlmostEqual(hits[0].score, 2 / 61)
        self.assertEqual((hits[0].retrieval_method, hits[0].score_kind, hits[0].rank),
                         ("hybrid_rrf", "rrf", 1))
        self.assertEqual(candidate_depth(20), 60)
        class RecordingStore(VersionedRetrievalStore):
            calls = []

            def search(self, question, as_of, k):
                self.calls.append(("lexical", k))
                return super().search(question, as_of, k)

            def semantic_search(self, vector, provider, model, as_of, k):
                self.calls.append(("semantic", k))
                return super().semantic_search(vector, provider, model, as_of, k)

        VersionedRetriever(self.sessions, embedding_provider=self.provider,
                           store_factory=RecordingStore).search("harborplan", mode="hybrid", k=20)
        self.assertEqual(RecordingStore.calls, [("lexical", 60), ("semantic", 60)])
        self.assertLessEqual(len(hits), 3)
        self.assertNotIn(self.old.snapshot_id, {hit.snapshot_id for hit in hits})
        old_hits = self.retriever.search("heritageharbor", mode="hybrid", as_of=MIDDLE)
        self.assertIn(self.old.snapshot_id, {hit.snapshot_id for hit in old_hits})
        hit = hits[0]
        self.assertIsInstance(hit, Evidence)
        self.assertEqual((hit.source_key, hit.source_title, hit.canonical_url),
                         ("a", "Harbor guidance", "https://example.invalid/a"))
        self.assertEqual(hit.snapshot_content_sha256, self.new.content_sha256)
        self.assertEqual(hit.observed_at, NEW)
        self.assertEqual(hit.source_updated_at, NEW)
        self.assertEqual(hit.text_sha256, sha256_text(hit.text))
        self.assertEqual(hit.passage_ordinal, 0)
        self.assertIsInstance(hit.score, float)

    def test_08_real_postgres_through_http_query(self):
        answers = FakeAnswers()
        pipeline = QueryPipeline(self.retriever, EvidenceFirstAnswerService(answers))
        def status():
            with self.sessions() as session:
                return versioned_corpus_state(session, MODEL, 1, provider_configured=True)
        app.dependency_overrides[get_query_pipeline] = lambda: pipeline
        app.dependency_overrides[get_versioned_status] = status
        try:
            with TestClient(app) as client:
                response = client.post("/api/query", json={"q": "harborplan"})
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["status"], "answered")
                self.assertEqual(body["retrieval"]["retrieval_mode"], "hybrid")
                self.assertEqual(body["citations"][0]["snapshot_id"], str(self.new.snapshot_id))
                self.assertEqual(body["claims"][0]["citations"][0]["supporting_quote"], "Harborplan")
                self.assertEqual(body["citations"][0]["observed_at"], NEW.isoformat())
                self.assertEqual(answers.calls, 1)

                historical = client.post("/api/query", json={
                    "q": "heritageharbor", "retrieval_mode": "lexical", "as_of": MIDDLE.isoformat(),
                })
                self.assertEqual(historical.status_code, 200, historical.text)
                self.assertEqual(historical.json()["citations"][0]["snapshot_id"], str(self.old.snapshot_id))
                self.assertEqual(answers.calls, 2)

                chunks = client.post("/api/query", json={"q": "harborplan", "mode": "chunks"})
                self.assertEqual(chunks.status_code, 200, chunks.text)
                self.assertEqual(chunks.json()["status"], "retrieved")
                self.assertEqual(chunks.json()["chunks"][0]["snapshot_id"], str(self.new.snapshot_id))
                self.assertEqual(answers.calls, 2)

                absent = client.post("/api/query", json={"q": "unfindablezzzz", "retrieval_mode": "lexical"})
                self.assertEqual(absent.status_code, 200, absent.text)
                self.assertEqual(absent.json()["status"], "abstained")
                self.assertEqual(absent.json()["sources"], [])
                self.assertEqual(answers.calls, 2)
        finally:
            app.dependency_overrides.clear()


class IntegrationUrlGuardTests(unittest.TestCase):
    def test_rejects_normal_and_remote_database_urls(self):
        for url in (
            "postgresql+psycopg://app_user:app_password@localhost:5432/appdb",
            "postgresql+psycopg://rag_test:x@remote.example/rag_integration_test",
            "sqlite:///rag_integration_test",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                guarded_url(url)
