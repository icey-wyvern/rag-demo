"""HTTP integration of versioned retrieval and evidence-first generation."""

import json
import importlib.util
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy.exc import ProgrammingError

from api.evidence_answer import EvidenceFirstAnswerService
from api.main import app, get_query_pipeline, get_versioned_status
from api.query_service import QueryPipeline
from api.versioned_readiness import versioned_corpus_state
from api.versioned_retriever import SemanticRetrievalError
from tests.test_versioned_retrieval import evidence


def response(claims=None, *, status="answered", reason=None):
    return {"status": status, "claims": claims or [], "abstention_reason": reason}


def claim(text, evidence_id="E1", quote="Service users"):
    return {"text": text, "citations": [{"evidence_id": evidence_id, "supporting_quote": quote}]}


def ready(*, lexical=True, semantic=True):
    return {
        "status": "ready" if lexical and semantic else "running" if lexical else "pending",
        "ingest_version": 1, "documents_total": 1,
        "documents_ready": 1 if semantic else 0, "last_error": None,
        "is_ready": lexical and semantic,
        "versioned": {"sources": 1, "snapshots": 1, "current_passages": 1 if lexical else 0,
                      "embedded_passages": 1 if semantic else 0,
                      "lexical_ready": lexical, "semantic_ready": semantic},
    }


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []
        self.failure = None

    def search(self, question, *, mode, as_of, k):
        self.calls.append((question, mode, as_of, k))
        if self.failure:
            raise self.failure
        return self.results


class FakeProvider:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate(self, question, formatted_evidence, instructions):
        self.calls.append((question, formatted_evidence, instructions))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class QueryApiTests(unittest.TestCase):
    def setUp(self):
        self.first = evidence(text="Service users must be involved.")
        self.second = replace(
            evidence(text="Teams should review feedback."),
            source_id=UUID(int=81), snapshot_id=UUID(int=82), passage_id=UUID(int=83),
            source_key="synthetic:feedback", source_title="Feedback guidance",
            canonical_url="https://example.invalid/feedback", rank=2,
        )
        self.retriever = FakeRetriever([self.first])
        self.provider = FakeProvider(response([claim("Involve users.")]))
        self.pipeline = QueryPipeline(self.retriever, EvidenceFirstAnswerService(self.provider))
        self.status = ready()
        app.dependency_overrides[get_query_pipeline] = lambda: self.pipeline
        app.dependency_overrides[get_versioned_status] = lambda: self.status
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.client.close()

    def test_health_and_status_use_versioned_readiness_shape(self):
        self.status = ready(lexical=False, semantic=False)
        health = self.client.get("/health")
        status = self.client.get("/api/status")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["corpus"], self.status)
        self.assertEqual(status.json(), self.status)
        self.assertFalse(status.json()["is_ready"])

    def test_default_generate_is_hybrid_and_uses_real_validator(self):
        for module in ("api.ingest", "api.store", "api.vectorstore", "api.generator", "api.models"):
            self.assertIsNone(importlib.util.find_spec(module))
        resp = self.client.post("/api/query", json={"q": "  What should teams do?  "})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(self.retriever.calls, [("What should teams do?", "hybrid", None, 5)])
        self.assertEqual(body["status"], "answered")
        self.assertEqual(body["answer"], "Involve users.")
        self.assertEqual(body["citations"][0]["title"], self.first.source_title)
        self.assertEqual(body["citations"][0]["url"], self.first.canonical_url)
        self.assertEqual(body["sources"], body["citations"])
        self.assertEqual(body["claims"][0]["citations"][0]["supporting_quote"], "Service users")
        self.assertEqual(body["claims"][0]["citations"][0]["snapshot_id"], str(self.first.snapshot_id))
        self.assertEqual(body["claims"][0]["citations"][0]["passage_sha256"], self.first.text_sha256)
        self.assertEqual(body["retrieval"], {"retrieval_mode": "hybrid", "k": 5, "as_of": None, "evidence_count": 1})
        self.assertEqual(body["chunks"][0]["snippet"], self.first.text)
        self.assertEqual(json.loads(self.provider.calls[0][1])[0]["passage_text"], self.first.text)

    def test_explicit_lexical_semantic_and_question_alias(self):
        for mode in ("lexical", "semantic"):
            with self.subTest(mode=mode):
                resp = self.client.post("/api/query", json={"question": "Question", "retrieval_mode": mode, "k": 3})
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(self.retriever.calls[-1], ("Question", mode, None, 3))
                self.assertEqual(resp.json()["retrieval"]["retrieval_mode"], mode)

    def test_as_of_preserves_timezone_and_versioned_provenance(self):
        observed = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
        old = replace(self.first, snapshot_id=UUID(int=91), observed_at=observed,
                      snapshot_content_sha256="a" * 64)
        self.retriever.results = [old]
        cutoff = "2026-10-15T13:00:00+00:00"
        resp = self.client.post("/api/query", json={"q": "Question", "as_of": cutoff})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.retriever.calls[-1][2], datetime.fromisoformat(cutoff))
        self.assertEqual(resp.json()["citations"][0]["snapshot_id"], str(old.snapshot_id))
        self.assertEqual(resp.json()["citations"][0]["observed_at"], observed.isoformat())
        self.assertEqual(resp.json()["retrieval"]["as_of"], cutoff)

    def test_invalid_request_values_are_rejected_before_retrieval(self):
        payloads = [
            {"q": "Question", "as_of": "2026-10-15T13:00:00"},
            {"q": "Question", "retrieval_mode": "vector"},
            {"q": "Question", "mode": "unknown"},
            {"q": "Question", "k": 0},
            {"q": " "},
            {"q": "A", "question": "B"},
            {"q": "Question", "unexpected_option": True},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/query", json=payload).status_code, 422)
        self.assertEqual(self.retriever.calls, [])

    def test_multiple_claims_dedupe_source_links_from_validated_citations(self):
        self.retriever.results = [self.first, self.second]
        self.provider.output = response([
            claim("Involve users."), claim("Involve users again."),
            claim("Review feedback.", "E2", "review feedback"),
        ])
        resp = self.client.post("/api/query", json={"q": "Question"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["claims"]), 3)
        self.assertEqual(len(body["citations"]), 2)
        self.assertEqual([item["source_key"] for item in body["citations"]],
                         [self.first.source_key, self.second.source_key])

    def test_uncited_retrieved_passage_is_not_a_source_link(self):
        self.retriever.results = [self.first, self.second]
        body = self.client.post("/api/query", json={"q": "Question"}).json()
        self.assertEqual(len(body["chunks"]), 2)
        self.assertEqual(len(body["citations"]), 1)
        self.assertEqual(body["citations"][0]["passage_id"], str(self.first.passage_id))

    def test_no_evidence_abstains_without_provider_call(self):
        self.retriever.results = []
        body = self.client.post("/api/query", json={"q": "Question"}).json()
        self.assertEqual(body["status"], "abstained")
        self.assertIn("Insufficient retrieved evidence", body["answer"])
        self.assertEqual(body["citations"], [])
        self.assertEqual(body["sources"], [])
        self.assertEqual(body["chunks"], [])
        self.assertEqual(self.provider.calls, [])

    def test_model_requested_abstention_is_success_with_no_sources(self):
        self.provider.output = response(status="abstained", reason="The passages do not answer this.")
        resp = self.client.post("/api/query", json={"q": "Question"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["answer"], "The passages do not answer this.")
        self.assertEqual(resp.json()["citations"], [])

    def test_contract_failures_return_no_invalid_answer(self):
        for citation in (
            {"evidence_id": "E99", "supporting_quote": "Service users"},
            {"evidence_id": "E1", "supporting_quote": "Invented quotation"},
        ):
            self.provider.output = response([{"text": "Private invalid answer", "citations": [citation]}])
            with self.subTest(citation=citation):
                resp = self.client.post("/api/query", json={"q": "Question"})
                self.assertEqual(resp.status_code, 502)
                self.assertNotIn("Private invalid answer", resp.text)
                self.assertEqual(resp.json()["detail"], "Generated answer failed evidence validation.")

    def test_chunks_mode_uses_evidence_and_skips_generation(self):
        resp = self.client.post("/api/query", json={"q": "Question", "mode": "chunks", "retrieval_mode": "lexical"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "retrieved")
        self.assertEqual(body["chunks"][0]["text"], self.first.text)
        self.assertEqual(body["chunks"][0]["snapshot_id"], str(self.first.snapshot_id))
        self.assertEqual(body["citations"], [])
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.retriever.calls[-1][1], "lexical")

    def test_readiness_blocks_missing_corpus_or_semantic_index_but_allows_lexical(self):
        self.status = ready(lexical=False, semantic=False)
        resp = self.client.post("/api/query", json={"q": "Question"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("Versioned corpus", resp.json()["detail"])
        self.status = ready(lexical=True, semantic=False)
        resp = self.client.post("/api/query", json={"q": "Question"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("Semantic index unavailable", resp.json()["detail"])
        resp = self.client.post("/api/query", json={"q": "Question", "retrieval_mode": "lexical", "mode": "chunks"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(self.retriever.calls), 1)
        self.status = ready()
        self.status["versioned"]["provider_configured"] = False
        resp = self.client.post("/api/query", json={"q": "Question"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("provider is not configured", resp.json()["detail"])

    def test_retriever_and_provider_failures_are_sanitized(self):
        self.retriever.failure = SemanticRetrievalError("private model and key")
        resp = self.client.post("/api/query", json={"q": "Question"})
        self.assertEqual(resp.status_code, 503)
        self.assertNotIn("private", resp.text)
        self.retriever.failure = RuntimeError("database password internal")
        resp = self.client.post("/api/query", json={"q": "Question"})
        self.assertEqual(resp.status_code, 503)
        self.assertNotIn("password", resp.text)
        self.retriever.failure = None
        self.provider.output = RuntimeError("API key internal")
        resp = self.client.post("/api/query", json={"q": "Question"})
        self.assertEqual(resp.status_code, 503)
        self.assertNotIn("API key", resp.text)


class VersionedReadinessTests(unittest.TestCase):
    def test_counts_use_current_version_and_configured_embedding_identity(self):
        for row, expected in [
            ({"source_count": 0, "snapshot_count": 0, "current_passages": 0,
              "embedded_passages": 0, "embedded_sources": 0}, ("pending", False, False)),
            ({"source_count": 1, "snapshot_count": 2, "current_passages": 3,
              "embedded_passages": 0, "embedded_sources": 0}, ("running", True, False)),
            ({"source_count": 1, "snapshot_count": 2, "current_passages": 3,
              "embedded_passages": 2, "embedded_sources": 1}, ("ready", True, True)),
        ]:
            with self.subTest(row=row):
                session = Mock()
                session.execute.return_value.mappings.return_value.one.return_value = row
                status = versioned_corpus_state(session, "fixture-model", 7)
                self.assertEqual(status["status"], expected[0])
                self.assertEqual(status["versioned"]["lexical_ready"], expected[1])
                self.assertEqual(status["versioned"]["semantic_ready"], expected[2])
                self.assertEqual(status["is_ready"], expected[1] and expected[2])
                statement, params = session.execute.call_args.args
                self.assertIn("row_number() OVER", str(statement))
                self.assertIn("pe.provider = 'openai' AND pe.model = :model", str(statement))
                self.assertEqual(params, {"model": "fixture-model"})

    def test_missing_provider_configuration_keeps_default_hybrid_not_ready(self):
        session = Mock()
        session.execute.return_value.mappings.return_value.one.return_value = {
            "source_count": 1, "snapshot_count": 1, "current_passages": 2,
            "embedded_passages": 2, "embedded_sources": 1,
        }
        status = versioned_corpus_state(session, "fixture-model", 1, provider_configured=False)
        self.assertEqual(status["status"], "error")
        self.assertFalse(status["is_ready"])
        self.assertTrue(status["versioned"]["lexical_ready"])
        self.assertFalse(status["versioned"]["provider_configured"])

    def test_missing_versioned_tables_are_pending_without_hiding_other_sql_errors(self):
        class DatabaseError(Exception):
            def __init__(self, sqlstate):
                self.sqlstate = sqlstate

        session = Mock()
        session.execute.side_effect = ProgrammingError("query", {}, DatabaseError("42P01"))
        pending = versioned_corpus_state(session, "fixture-model", 1)
        self.assertEqual(pending["status"], "pending")
        self.assertFalse(pending["versioned"]["schema_ready"])
        session.execute.side_effect = ProgrammingError("query", {}, DatabaseError("42501"))
        with self.assertRaises(ProgrammingError):
            versioned_corpus_state(session, "fixture-model", 1)


if __name__ == "__main__":
    unittest.main()
