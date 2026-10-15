"""Deterministic hostile-content regression cases for the internal answer boundary."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal
from uuid import UUID

from .evidence import Evidence
from .evidence_answer import AnswerError, EvidenceFirstAnswerService, failure_category


FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "hostile_evidence_cases.json"
Outcome = Literal["accepted", "rejected", "abstained"]


@dataclass(frozen=True)
class SecurityCase:
    case_id: str
    question: str
    evidence: tuple[str, ...]
    provider_response: object
    expected_outcome: Outcome
    expected_category: str | None = None


@dataclass(frozen=True)
class SecurityEvaluationSummary:
    cases: int
    passed: int
    failed: int
    failed_case_ids: tuple[str, ...]
    failure_categories: dict[str, int]


def load_cases(path: Path = FIXTURE_PATH) -> tuple[SecurityCase, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(SecurityCase(
        case_id=item["case_id"], question=item["question"], evidence=tuple(item["evidence"]),
        provider_response=item["provider_response"], expected_outcome=item["expected_outcome"],
        expected_category=item.get("expected_category"),
    ) for item in raw)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Security case IDs must be unique")
    if any(case.expected_outcome not in ("accepted", "rejected", "abstained") for case in cases):
        raise ValueError("Invalid expected security outcome")
    return cases


def _evidence(case: SecurityCase) -> list[Evidence]:
    observed = datetime(2026, 10, 15, tzinfo=timezone.utc)
    return [Evidence(
        source_id=UUID(int=1), source_key="synthetic:security", source_title="Synthetic hostile-content fixture",
        canonical_url="https://example.invalid/security-fixture", snapshot_id=UUID(int=2),
        snapshot_content_sha256="0" * 64, retrieved_url="https://example.invalid/security-fixture",
        observed_at=observed, source_published_at=None, source_updated_at=None,
        passage_id=UUID(int=index), passage_ordinal=index - 1, page=None, text=text,
        text_sha256=sha256(text.encode("utf-8")).hexdigest(), retrieval_method="security_fixture",
        score=0.0, rank=index, score_kind="not_ranked",
    ) for index, text in enumerate(case.evidence, 1)]


class _FakeProvider:
    def __init__(self, response: object):
        self.response = response

    def generate(self, question: str, formatted_evidence: str, instructions: str) -> object:
        return self.response


def evaluate_cases(cases: tuple[SecurityCase, ...]) -> SecurityEvaluationSummary:
    failed = []
    categories: dict[str, int] = {}
    for case in cases:
        try:
            answer = EvidenceFirstAnswerService(_FakeProvider(case.provider_response)).answer(
                case.question, _evidence(case),
            )
            outcome: Outcome = "abstained" if answer.status == "abstained" else "accepted"
            category = None
        except AnswerError as error:
            outcome = "rejected"
            category = failure_category(error)
            categories[category] = categories.get(category, 0) + 1
        if outcome != case.expected_outcome or category != case.expected_category:
            failed.append(case.case_id)
    return SecurityEvaluationSummary(
        cases=len(cases), passed=len(cases) - len(failed), failed=len(failed),
        failed_case_ids=tuple(failed), failure_categories=categories,
    )


if __name__ == "__main__":
    from dataclasses import asdict

    summary = evaluate_cases(load_cases())
    print(json.dumps(asdict(summary), sort_keys=True))
    raise SystemExit(1 if summary.failed else 0)
