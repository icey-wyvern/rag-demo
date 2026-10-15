"""Deterministic, retrieval-only evaluation of the versioned Evidence contract."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Protocol

from .evidence import Evidence


class Retriever(Protocol):
    def search(self, question: str, *, as_of: datetime | None = None, k: int = 5) -> list[Evidence]: ...


class ModeRetriever(Protocol):
    def search(self, question: str, *, mode: str = "lexical", as_of: datetime | None = None,
               k: int = 5) -> list[Evidence]: ...


@dataclass(frozen=True)
class ExpectedEvidence:
    source_key: str
    snapshot_sha256: str | None = None
    required_terms: tuple[str, ...] = ()

    def matches(self, evidence: Evidence) -> bool:
        return (
            evidence.source_key == self.source_key
            and (self.snapshot_sha256 is None or evidence.snapshot_content_sha256 == self.snapshot_sha256)
            and all(term.casefold() in evidence.text.casefold() for term in self.required_terms)
        )


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    as_of: datetime | None
    expected: tuple[ExpectedEvidence, ...]
    k: int = 3


@dataclass(frozen=True)
class EvalSummary:
    cases: int
    positive_cases: int
    no_evidence_cases: int
    recall_at_1: float
    recall_at_3: float
    mrr: float
    temporal_cases: int
    temporal_accuracy: float | None
    no_evidence_accuracy: float | None
    failed_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModeComparison:
    summaries: dict[str, EvalSummary]
    versus_lexical: dict[str, dict[str, str]]


def load_cases(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("evaluation dataset must be a non-empty JSON array")
    cases = []
    seen = set()
    for row in data:
        if not isinstance(row, dict) or set(row) - {"id", "query", "as_of", "expected", "k"}:
            raise ValueError("evaluation case has invalid fields")
        case_id = row.get("id")
        query = row.get("query")
        raw_expected = row.get("expected")
        k = row.get("k", 3)
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("evaluation case ID is blank or duplicated")
        seen.add(case_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{case_id}: query must be non-empty")
        if isinstance(k, bool) or not isinstance(k, int) or not 3 <= k <= 20:
            raise ValueError(f"{case_id}: k must be between 3 and 20")
        as_of = None
        if row.get("as_of") is not None:
            try:
                as_of = datetime.fromisoformat(row["as_of"].replace("Z", "+00:00"))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"{case_id}: invalid as_of") from exc
            if as_of.tzinfo is None or as_of.utcoffset() is None:
                raise ValueError(f"{case_id}: as_of must be timezone-aware")
        if not isinstance(raw_expected, list):
            raise ValueError(f"{case_id}: expected must be a list; [] means no evidence")
        expected = []
        expected_keys = set()
        for item in raw_expected:
            if not isinstance(item, dict) or set(item) - {"source_key", "snapshot_sha256", "required_terms"}:
                raise ValueError(f"{case_id}: invalid expected evidence")
            source_key = item.get("source_key")
            digest = item.get("snapshot_sha256")
            terms = item.get("required_terms", [])
            if not isinstance(source_key, str) or not source_key.strip():
                raise ValueError(f"{case_id}: expected source_key is required")
            if source_key in expected_keys:
                raise ValueError(f"{case_id}: duplicate expected source_key")
            expected_keys.add(source_key)
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError(f"{case_id}: invalid snapshot_sha256")
            if not isinstance(terms, list) or any(not isinstance(t, str) or not t.strip() for t in terms):
                raise ValueError(f"{case_id}: invalid required_terms")
            expected.append(ExpectedEvidence(source_key, digest, tuple(terms)))
        if as_of is not None and expected and any(e.snapshot_sha256 is None for e in expected):
            raise ValueError(f"{case_id}: temporal expected evidence requires snapshot_sha256")
        cases.append(EvalCase(case_id, query, as_of, tuple(expected), k))
    return cases


def evaluate(retriever: Retriever, cases: list[EvalCase]) -> EvalSummary:
    if not cases:
        raise ValueError("evaluation cases must not be empty")
    positive = no_evidence = recall_1 = recall_3 = temporal = temporal_correct = no_evidence_correct = 0
    reciprocal_sum = 0.0
    failed = []
    for case in cases:
        results = retriever.search(case.query, as_of=case.as_of, k=case.k)
        if not case.expected:
            no_evidence += 1
            correct = not results
            no_evidence_correct += int(correct)
        else:
            positive += 1
            first = next((index for index, evidence in enumerate(results, 1)
                          if any(expected.matches(evidence) for expected in case.expected)), None)
            recall_1 += int(first == 1)
            recall_3 += int(first is not None and first <= 3)
            reciprocal_sum += 1 / first if first else 0.0
            correct = first is not None and first <= case.k
        if case.as_of is not None:
            temporal += 1
            if case.expected:
                # The expected version must be found, and no result for that
                # source may come from another snapshot.
                relevant_keys = {item.source_key for item in case.expected}
                correct_version = correct and all(
                    any(item.source_key == evidence.source_key
                        and item.snapshot_sha256 == evidence.snapshot_content_sha256
                        for item in case.expected)
                    for evidence in results if evidence.source_key in relevant_keys
                )
            else:
                correct_version = correct
            temporal_correct += int(correct_version)
            correct = correct and correct_version
        if not correct:
            failed.append(case.case_id)
    return EvalSummary(
        cases=len(cases), positive_cases=positive, no_evidence_cases=no_evidence,
        recall_at_1=recall_1 / positive if positive else 0.0,
        recall_at_3=recall_3 / positive if positive else 0.0,
        mrr=reciprocal_sum / positive if positive else 0.0,
        temporal_cases=temporal,
        temporal_accuracy=temporal_correct / temporal if temporal else None,
        no_evidence_accuracy=no_evidence_correct / no_evidence if no_evidence else None,
        failed_case_ids=tuple(failed),
    )


def evaluate_mode(retriever: ModeRetriever, cases: list[EvalCase], mode: str) -> EvalSummary:
    if mode not in ("lexical", "semantic", "hybrid"):
        raise ValueError("Unknown retrieval mode")

    class SelectedMode:
        def search(self, question: str, *, as_of: datetime | None = None, k: int = 5) -> list[Evidence]:
            return retriever.search(question, mode=mode, as_of=as_of, k=k)

    return evaluate(SelectedMode(), cases)


def compare_modes(retriever: ModeRetriever, cases: list[EvalCase]) -> ModeComparison:
    summaries = {mode: evaluate_mode(retriever, cases, mode)
                 for mode in ("lexical", "semantic", "hybrid")}
    metrics = ("recall_at_1", "recall_at_3", "mrr", "temporal_accuracy", "no_evidence_accuracy")
    baseline = summaries["lexical"]
    comparisons = {}
    for mode in ("semantic", "hybrid"):
        comparisons[mode] = {}
        for metric in metrics:
            base_value = getattr(baseline, metric)
            value = getattr(summaries[mode], metric)
            if base_value is None or value is None:
                status = "unavailable"
            elif abs(value - base_value) < 1e-12:
                status = "match"
            elif value > base_value:
                status = "improve"
            else:
                status = "regress"
            comparisons[mode][metric] = status
    return ModeComparison(summaries=summaries, versus_lexical=comparisons)
