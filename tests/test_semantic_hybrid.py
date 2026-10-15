"""Offline semantic, temporal, RRF, and mode-comparison tests."""

import math
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import UUID

from pgvector.sqlalchemy import Vector

from api.hybrid_rrf import candidate_depth, fuse_rrf
from api.retrieval_eval import EvalCase, ExpectedEvidence, compare_modes
from api.versioned_retrieval_store import VersionedRetrievalStore, eligible_snapshots_sql, semantic_statement
from api.versioned_retriever import SemanticRetrievalError, VersionedRetriever
from tests.test_versioned_retrieval import evidence


OLD = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
NEW = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)


def vector(x, y=0.0):
    return [float(x), float(y)] + [0.0] * 1534


class FakeProvider:
    provider_id = "fixture"
    model_id = "fixture-v1"
    dimensions = 1536

    def __init__(self, response=None):
        self.response = [vector(1)] if response is None else response
        self.calls = []
        self.failure = None

    def embed_texts(self, texts):
        self.calls.append(tuple(texts))
        if self.failure:
            raise self.failure
        return self.response


class Sessions:
    def __call__(self):
        class Context:
            def __enter__(self): return object()
            def __exit__(self, *_): return False
        return Context()


class FakeStore:
    """Known-vector fake of the SQL eligibility and cosine ordering contract."""

    rows = [
        ("a", OLD, UUID(int=1), "old-a", vector(1), "fixture", "fixture-v1"),
        ("a", NEW, UUID(int=2), "new-a", vector(0, 1), "fixture", "fixture-v1"),
        ("b", OLD, UUID(int=3), "source-b", vector(0.8, 0.6), "fixture", "fixture-v1"),
        ("b", OLD, UUID(int=4), "wrong-model", vector(1), "fixture", "fixture-v2"),
    ]

    def __init__(self, session):
        self.calls = []
        self.available = True
        self.consistent_read_started = False

    def begin_consistent_read(self):
        self.consistent_read_started = True

    def has_embeddings(self, provider, model):
        return self.available and any(row[5:7] == (provider, model) for row in self.rows)

    def _eligible(self, as_of):
        latest = {}
        for source, observed, passage_id, name, embedding, provider, model in self.rows:
            if as_of is None or observed <= as_of:
                if source not in latest or observed > latest[source]:
                    latest[source] = observed
        return latest

    def semantic_search(self, query_vector, provider, model, as_of, k):
        self.calls.append(("semantic", k, provider, model, as_of))
        eligible = self._eligible(as_of)
        hits = []
        for source, observed, passage_id, name, embedding, row_provider, row_model in self.rows:
            if eligible.get(source) != observed or (row_provider, row_model) != (provider, model):
                continue
            dot = sum(a * b for a, b in zip(query_vector, embedding))
            norm_a = math.sqrt(sum(x * x for x in query_vector))
            norm_b = math.sqrt(sum(x * x for x in embedding))
            distance = 1 - dot / (norm_a * norm_b)
            hit = replace(
                evidence(source, name.ljust(64, "x")[:64], text=name),
                passage_id=passage_id, observed_at=observed,
                score=distance, score_kind="cosine_distance",
                retrieval_method="pgvector_cosine_exact", lexical_rank=None,
            )
            hits.append(hit)
        hits.sort(key=lambda item: (item.score, item.source_key, item.passage_ordinal, item.passage_id))
        return [replace(hit, rank=rank, semantic_rank=rank) for rank, hit in enumerate(hits[:k], 1)]

    def search(self, query, as_of, k):
        self.calls.append(("lexical", k, None, None, as_of))
        # Only matching text in the eligible observation contributes.
        eligible = self._eligible(as_of)
        hits = [replace(evidence(source, name.ljust(64, "x")[:64], text=name),
                        passage_id=passage_id, observed_at=observed)
                for source, observed, passage_id, name, _, provider, model in self.rows
                if eligible.get(source) == observed and provider == "fixture" and model == "fixture-v1"
                and query.casefold() in name.casefold()]
        hits.sort(key=lambda item: (item.source_key, item.passage_id))
        return [replace(hit, rank=rank, lexical_rank=rank) for rank, hit in enumerate(hits[:k], 1)]


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.store = FakeStore(None)
        self.retriever = VersionedRetriever(
            Sessions(), embedding_provider=self.provider, store_factory=lambda _: self.store,
        )

    def test_semantic_current_as_of_and_provider_model_filter(self):
        current = self.retriever.search("question", mode="semantic")
        self.assertEqual([item.text for item in current], ["source-b", "new-a"])
        self.assertTrue(all(item.text not in ("old-a", "wrong-model") for item in current))
        self.assertEqual([item.rank for item in current], [1, 2])
        self.assertEqual([item.semantic_rank for item in current], [1, 2])
        self.assertLess(current[0].score, current[1].score)
        self.assertEqual(current[0].score_kind, "cosine_distance")
        self.assertEqual(current[0].retrieval_method, "pgvector_cosine_exact")
        older = self.retriever.search("question", mode="semantic", as_of=OLD)
        self.assertEqual([item.text for item in older], ["old-a", "source-b"])
        before = self.retriever.search("question", mode="semantic", as_of=datetime(2026, 10, 14, tzinfo=timezone.utc))
        self.assertEqual(before, [])
        self.assertEqual(self.provider.calls, [("question",)] * 3)

    def test_sql_uses_shared_temporal_eligibility_before_vector_ordering(self):
        current = str(semantic_statement(None))
        as_of = str(semantic_statement(OLD))
        shared = eligible_snapshots_sql(OLD)
        self.assertIn(shared, as_of)
        self.assertIn("ss.observed_at <= :as_of", as_of)
        self.assertNotIn("ss.observed_at <= :as_of", current)
        self.assertLess(as_of.index("eligible AS"), as_of.index("JOIN passage_embeddings"))
        self.assertIn("JOIN passages p ON p.snapshot_id = e.id", as_of)
        self.assertIn("pe.provider = :provider AND pe.model = :model", as_of)
        self.assertIn("pe.embedding <=> :query_vector AS score", as_of)
        self.assertIn("ORDER BY score ASC, s.source_key ASC, p.ordinal ASC, p.id ASC", as_of)
        self.assertEqual(semantic_statement(None)._bindparams["query_vector"].type.dim, 1536)
        self.assertIsInstance(semantic_statement(None)._bindparams["query_vector"].type, Vector)

    def test_store_maps_distance_without_calling_it_similarity(self):
        row = dict(vars(evidence()))
        for key in ("retrieval_method", "rank", "score_kind", "lexical_rank", "semantic_rank"):
            row.pop(key)
        row["score"] = 0.25
        class Result:
            def mappings(self): return self
            def all(self): return [row]
        class Session:
            def execute(self, statement, params):
                self.params = params
                return Result()
        session = Session()
        result = VersionedRetrievalStore(session).semantic_search(vector(1), "fixture", "fixture-v1", OLD, 3)
        self.assertEqual(result[0].score, 0.25)
        self.assertEqual(result[0].score_kind, "cosine_distance")
        self.assertEqual(result[0].rank, 1)
        self.assertEqual(result[0].semantic_rank, 1)
        self.assertEqual(session.params["provider"], "fixture")
        self.assertEqual(session.params["model"], "fixture-v1")

    def test_hybrid_store_requests_one_repeatable_read_snapshot(self):
        session = Mock()
        VersionedRetrievalStore(session).begin_consistent_read()
        statement = session.execute.call_args.args[0]
        self.assertEqual(str(statement), "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

    def test_missing_provider_model_and_invalid_query_vectors_fail_before_search(self):
        self.store.available = False
        with self.assertRaisesRegex(SemanticRetrievalError, "No indexed passages"):
            self.retriever.search("question", mode="semantic")
        self.assertEqual(self.provider.calls, [])
        self.store.available = True
        for response, message in [
            ([], "0 vectors"),
            ([vector(1), vector(0, 1)], "2 vectors"),
            ([[0.1] * 3], "1536 dimensions"),
            ([vector(math.nan)], "non-finite"),
            ([vector(0)], "zero vector"),
        ]:
            self.provider.response = response
            with self.subTest(message=message), self.assertRaisesRegex(SemanticRetrievalError, message):
                self.retriever.search("question", mode="semantic")
        self.assertEqual(self.store.calls, [])
        self.provider.failure = RuntimeError("provider offline")
        with self.assertRaisesRegex(SemanticRetrievalError, "provider failed"):
            self.retriever.search("question", mode="semantic")
        with self.assertRaisesRegex(SemanticRetrievalError, "requires an embedding provider"):
            VersionedRetriever(Sessions(), store_factory=lambda _: self.store).search("question", mode="semantic")
        wrong_size = FakeProvider()
        wrong_size.dimensions = 3
        with self.assertRaisesRegex(ValueError, "1536"):
            VersionedRetriever(Sessions(), embedding_provider=wrong_size)

    def test_hybrid_uses_bounded_deeper_candidates(self):
        result = self.retriever.search("source", mode="hybrid", k=2)
        self.assertEqual(self.store.calls[-2][0:2], ("lexical", 6))
        self.assertEqual(self.store.calls[-1][0:2], ("semantic", 6))
        self.assertLessEqual(len(result), 2)
        self.assertTrue(all(item.retrieval_method == "hybrid_rrf" for item in result))
        self.assertEqual(len(self.provider.calls), 1)
        self.assertTrue(self.store.consistent_read_started)

    def test_hybrid_respects_current_and_as_of_versions(self):
        current = self.retriever.search("old-a", mode="hybrid", k=3)
        self.assertTrue(all(item.text != "old-a" for item in current))
        older = self.retriever.search("old-a", mode="hybrid", as_of=OLD, k=3)
        self.assertEqual(older[0].text, "old-a")
        self.assertEqual((older[0].lexical_rank, older[0].semantic_rank), (1, 1))


class RrfTests(unittest.TestCase):
    def test_overlap_and_one_sided_results(self):
        a = replace(evidence("a"), passage_id=UUID(int=1))
        b = replace(evidence("b"), passage_id=UUID(int=2))
        c = replace(evidence("c"), passage_id=UUID(int=3))
        fused = fuse_rrf([a, b], [b, c], k=3, constant=60)
        self.assertEqual([item.passage_id for item in fused], [UUID(int=2), UUID(int=1), UUID(int=3)])
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 61)
        self.assertEqual((fused[0].lexical_rank, fused[0].semantic_rank), (2, 1))
        self.assertEqual((fused[1].lexical_rank, fused[1].semantic_rank), (1, None))
        self.assertEqual((fused[2].lexical_rank, fused[2].semantic_rank), (None, 2))
        self.assertEqual([item.rank for item in fused], [1, 2, 3])
        self.assertTrue(all(item.score_kind == "rrf" for item in fused))

    def test_ties_are_deterministic_and_duplicate_input_passages_not_double_counted(self):
        a = replace(evidence("a"), passage_id=UUID(int=1))
        b = replace(evidence("b"), passage_id=UUID(int=2))
        first = fuse_rrf([b, a, a], [a, b], k=2)
        second = fuse_rrf([b, a, a], [a, b], k=2)
        self.assertEqual([item.source_key for item in first], ["a", "b"])
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0].score, 1 / 62 + 1 / 61)

    def test_candidate_depth_and_mode_validation(self):
        self.assertEqual((candidate_depth(1), candidate_depth(5), candidate_depth(20)), (3, 15, 60))
        with self.assertRaisesRegex(ValueError, "RRF constant"):
            fuse_rrf([], [], k=1, constant=0)
        retriever = VersionedRetriever(Sessions())
        with self.assertRaisesRegex(ValueError, "mode must"):
            retriever.search("question", mode="unknown")
        with self.assertRaisesRegex(ValueError, "k must"):
            retriever.search("question", mode="hybrid", k=21)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            retriever.search("question", mode="semantic", as_of=datetime(2026, 10, 15))


class ComparisonTests(unittest.TestCase):
    def test_modes_use_same_cases_and_report_improve_match_regress(self):
        cases = [
            EvalCase("hit", "q1", None, (ExpectedEvidence("right"),)),
            EvalCase("temporal", "q2", OLD, (ExpectedEvidence("right", "a" * 64),)),
            EvalCase("empty", "q3", None, ()),
        ]
        right = evidence("right", "a" * 64)
        wrong = evidence("wrong", "b" * 64)
        outputs = {
            "lexical": {"q1": [right], "q2": [], "q3": []},
            "semantic": {"q1": [wrong, right], "q2": [right], "q3": [wrong]},
            "hybrid": {"q1": [right], "q2": [right], "q3": []},
        }
        class Fake:
            def search(self, question, *, mode, as_of, k):
                return outputs[mode][question]
        comparison = compare_modes(Fake(), cases)
        self.assertEqual(set(comparison.summaries), {"lexical", "semantic", "hybrid"})
        self.assertEqual(comparison.summaries["lexical"].recall_at_1, 0.5)
        self.assertEqual(comparison.summaries["semantic"].recall_at_3, 1.0)
        self.assertEqual(comparison.summaries["hybrid"].mrr, 1.0)
        self.assertEqual(comparison.versus_lexical["semantic"]["recall_at_1"], "match")
        self.assertEqual(comparison.versus_lexical["semantic"]["recall_at_3"], "improve")
        self.assertEqual(comparison.versus_lexical["semantic"]["no_evidence_accuracy"], "regress")
        self.assertEqual(comparison.versus_lexical["hybrid"]["temporal_accuracy"], "improve")
        self.assertEqual(comparison.versus_lexical["hybrid"]["no_evidence_accuracy"], "match")
        self.assertEqual(comparison.summaries["lexical"].failed_case_ids, ("temporal",))
        self.assertEqual(comparison.summaries["semantic"].failed_case_ids, ("empty",))


if __name__ == "__main__":
    unittest.main()
