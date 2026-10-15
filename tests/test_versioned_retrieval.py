import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from api.evidence import Evidence
from api.retrieval_eval import EvalCase, ExpectedEvidence, evaluate, load_cases
from api.versioned_models import Passage
from api.versioned_retrieval_store import VersionedRetrievalStore, lexical_statement
from api.versioned_retriever import VersionedRetriever


T1 = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
T2 = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
DATASET = Path(__file__).parent.parent / "eval" / "fixture_retrieval_cases.json"


def evidence(source="govuk:one", snapshot="a" * 64, *, rank=1, score=0.5, text="service users"):
    return Evidence(
        source_id=UUID(int=1), source_key=source, source_title="A source",
        canonical_url="https://www.gov.uk/example", snapshot_id=UUID(int=2),
        snapshot_content_sha256=snapshot, retrieved_url="https://www.gov.uk/api/content/example", observed_at=T1,
        source_published_at=None, source_updated_at=None, passage_id=UUID(int=3),
        passage_ordinal=0, page=None, text=text, text_sha256="c" * 64,
        retrieval_method="postgresql_fts_english", score=score, rank=rank,
    )


class _Sessions:
    def __call__(self):
        class Context:
            def __enter__(self): return object()
            def __exit__(self, *_): return False
        return Context()


class _FakeTemporalStore:
    """Service contract fake; SQL structure is checked separately below."""
    observations = [
        ("govuk:one", T1, UUID(int=2), "a" * 64, "legislative constraints"),
        ("govuk:one", T2, UUID(int=4), "b" * 64, "emerging technologies AI"),
        ("govuk:two", T1, UUID(int=5), "d" * 64, "service users"),
    ]

    def __init__(self, session):
        pass

    def search(self, question, as_of, k):
        eligible = {}
        for source, observed, snapshot_id, digest, passage_text in self.observations:
            if as_of is None or observed <= as_of:
                if source not in eligible or (observed, snapshot_id) > (eligible[source][0], eligible[source][1]):
                    eligible[source] = (observed, snapshot_id, digest, passage_text)
        hits = []
        for source, (observed, snapshot_id, digest, passage_text) in eligible.items():
            if question.casefold() in passage_text.casefold() or question == "all":
                hits.append(replace(evidence(source, digest, text=passage_text), snapshot_id=snapshot_id, observed_at=observed))
        hits.sort(key=lambda e: (-e.score, e.source_key, e.passage_ordinal, e.passage_id))
        return [replace(hit, rank=rank) for rank, hit in enumerate(hits[:k], 1)]


class TemporalRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = VersionedRetriever(_Sessions(), store_factory=_FakeTemporalStore)

    def test_current_and_as_of_select_only_latest_eligible_snapshot(self):
        current = self.retriever.search("all")
        self.assertEqual([(e.source_key, e.snapshot_content_sha256) for e in current],
                         [("govuk:one", "b" * 64), ("govuk:two", "d" * 64)])
        old = self.retriever.search("all", as_of=T1)
        self.assertEqual([(e.source_key, e.snapshot_content_sha256) for e in old],
                         [("govuk:one", "a" * 64), ("govuk:two", "d" * 64)])
        self.assertEqual(self.retriever.search("all", as_of=datetime(2026, 10, 14, tzinfo=timezone.utc)), [])
        self.assertEqual(self.retriever.search("legislative constraints"), [])
        self.assertEqual([e.source_key for e in self.retriever.search("legislative constraints", as_of=T1)], ["govuk:one"])

    def test_equal_scores_have_stable_order_and_contract_is_immutable(self):
        first = self.retriever.search("all")
        self.assertEqual([e.source_key for e in first], ["govuk:one", "govuk:two"])
        self.assertEqual([e.rank for e in first], [1, 2])
        self.assertTrue(all(type(e) is Evidence and e.retrieval_method == "postgresql_fts_english" for e in first))
        with self.assertRaises(FrozenInstanceError):
            first[0].score = 2.0

    def test_input_validation(self):
        for k in (0, 21, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "k must"):
                self.retriever.search("all", k=k)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.retriever.search("all", as_of=datetime(2026, 10, 15))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.retriever.search("   ")

    def test_postgres_query_selects_snapshot_before_matching_and_has_stable_order(self):
        current = str(lexical_statement(None))
        temporal = str(lexical_statement(T1))
        self.assertIn("row_number() OVER", current)
        self.assertIn("PARTITION BY ss.source_id", current)
        self.assertIn("ORDER BY ss.observed_at DESC, ss.created_at DESC, ss.id DESC", current)
        self.assertIn("JOIN passages p ON p.snapshot_id = e.id", current)
        self.assertLess(current.index("WITH eligible"), current.index("JOIN passages"))
        self.assertIn("e.version_rank = 1", current)
        self.assertIn("ss.observed_at <= :as_of", temporal)
        self.assertNotIn("ss.observed_at <= :as_of", current)
        self.assertIn("to_tsvector('english', p.text) @@ search_query.tsq", current)
        self.assertIn("websearch_to_tsquery('english', :question)", current)
        self.assertIn("ORDER BY score DESC, s.source_key ASC, p.ordinal ASC, p.id ASC", current)
        indexes = [index for index in Passage.__table__.indexes if index.name == "ix_passages_text_english_fts"]
        self.assertEqual(len(indexes), 1)
        compiled = str(CreateIndex(indexes[0]).compile(dialect=postgresql.dialect()))
        self.assertIn("USING gin (to_tsvector('english', text))", compiled)

    def test_store_maps_rows_to_evidence_without_orm_objects(self):
        row = dict(vars(evidence()))
        row.pop("retrieval_method")
        row.pop("rank")
        row["score"] = 0.75
        class Result:
            def mappings(self): return self
            def all(self): return [row]
        class Session:
            def execute(self, statement, params):
                self.params = params
                return Result()
        session = Session()
        result = VersionedRetrievalStore(session).search("service", T1, 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_key, "govuk:one")
        self.assertEqual(result[0].snapshot_content_sha256, "a" * 64)
        self.assertEqual(result[0].retrieved_url, "https://www.gov.uk/api/content/example")
        self.assertEqual(result[0].score, 0.75)
        self.assertEqual(result[0].rank, 1)
        self.assertEqual(session.params, {"question": "service", "as_of": T1, "k": 3})


class EvaluationTests(unittest.TestCase):
    def test_dataset_is_reviewable_and_contains_required_scenarios(self):
        cases = load_cases(DATASET)
        self.assertEqual(len(cases), 7)
        self.assertEqual(sum(c.as_of is not None for c in cases), 2)
        self.assertEqual(sum(not c.expected for c in cases), 2)
        self.assertTrue(any(len(c.expected) > 1 for c in cases))
        self.assertEqual(len(load_cases(DATASET.parent / "current_retrieval_cases.json")), 3)

    def test_metrics_failures_and_no_evidence_rule(self):
        cases = [
            EvalCase("first", "q1", None, (ExpectedEvidence("a"),)),
            EvalCase("second", "q2", None, (ExpectedEvidence("b"),)),
            EvalCase("missing", "q3", None, (ExpectedEvidence("c"),)),
            EvalCase("old", "q4", T1, (ExpectedEvidence("a", "a" * 64),)),
            EvalCase("empty", "q5", None, ()),
            EvalCase("false-hit", "q6", T1, ()),
        ]
        responses = {
            "q1": [evidence("a")],
            "q2": [evidence("x"), evidence("b", rank=2)],
            "q3": [],
            "q4": [evidence("a", "b" * 64)],
            "q5": [],
            "q6": [evidence("a")],
        }
        class Fake:
            def search(self, question, *, as_of=None, k=5): return responses[question]
        summary = evaluate(Fake(), cases)
        self.assertEqual(summary.cases, 6)
        self.assertEqual(summary.positive_cases, 4)
        self.assertEqual(summary.no_evidence_cases, 2)
        self.assertEqual(summary.recall_at_1, 0.25)
        self.assertEqual(summary.recall_at_3, 0.5)
        self.assertEqual(summary.mrr, 0.375)
        self.assertEqual(summary.temporal_accuracy, 0.0)
        self.assertEqual(summary.no_evidence_accuracy, 0.5)
        self.assertEqual(summary.failed_case_ids, ("missing", "old", "false-hit"))

    def test_invalid_dataset_fails_clearly(self):
        template = {"id": "case", "query": "service", "expected": []}
        variants = [
            ([template, template], "duplicated"),
            ([{**template, "as_of": "2026-01-01T00:00:00"}], "timezone-aware"),
            ([{**template, "expected": None}], "expected must be a list"),
            ([{**template, "expected": [{"source_key": "x"}], "as_of": "2026-01-01T00:00:00Z"}], "requires snapshot_sha256"),
            ([{**template, "expected": [{"source_key": "x", "snapshot_sha256": "bad"}]}], "invalid snapshot_sha256"),
        ]
        for rows, message in variants:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cases.json"
                path.write_text(json.dumps(rows), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_cases(path)


if __name__ == "__main__":
    unittest.main()
