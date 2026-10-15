"""Offline evaluation of answer behaviour given pre-supplied versioned Evidence."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Literal
from uuid import UUID

from .evidence import Evidence
from .evidence_answer import AnswerError, EvidenceFirstAnswerService, failure_category
from .security_eval import evidence_for_case, load_cases as load_security_cases
from .version_change import (
    PassageRef, SnapshotRef, SourceRef, VersionChangeSet,
    VersionChangeSummaryService, compare_passages,
)


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "eval" / "generation_cases.json"
Status = Literal["answered", "abstained", "rejected"]


@dataclass(frozen=True)
class EvidenceRequirement:
    source_key: str | None = None
    snapshot_id: UUID | None = None
    snapshot_hash: str | None = None
    passage_id: UUID | None = None
    passage_hash: str | None = None

    def matches(self, item: Evidence) -> bool:
        return all((wanted is None or actual == wanted) for wanted, actual in (
            (self.source_key, item.source_key),
            (self.snapshot_id, item.snapshot_id),
            (self.snapshot_hash, item.snapshot_content_sha256),
            (self.passage_id, item.passage_id),
            (self.passage_hash, item.text_sha256),
        ))


@dataclass(frozen=True)
class GenerationCase:
    case_id: str
    question: str
    evidence: tuple[Evidence, ...]
    provider_response: object
    expected_status: Status
    expected_category: str | None = None
    requirements: tuple[EvidenceRequirement, ...] = ()
    requirement_mode: Literal["all", "any"] = "all"
    expected_snapshot_ids: tuple[UUID, ...] = ()
    as_of: datetime | None = None
    change_set: VersionChangeSet | None = None
    expected_provider_calls: int = 1


@dataclass(frozen=True)
class CaseObservation:
    case_id: str
    observed_status: Status
    rejection_category: str | None
    provider_calls: int
    passed: bool
    diagnostics: tuple[str, ...]
    evidence_match: bool | None
    temporal_match: bool | None


@dataclass(frozen=True)
class GenerationEvaluationReport:
    cases: int
    passed: int
    failed: int
    failed_case_ids: tuple[str, ...]
    answer_status_accuracy: float | None
    abstention_accuracy: float | None
    correct_abstentions: int
    incorrect_abstentions: int
    incorrect_answers: int
    citation_coverage: float | None
    citation_validity: float | None
    expected_evidence_accuracy: float | None
    temporal_evidence_accuracy: float | None
    contract_pass_rate: float | None
    rejection_categories: dict[str, int]
    diagnostic_counts: dict[str, int]
    observations: tuple[CaseObservation, ...]


def _nonempty(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _digest(value, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value, label: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return result


def _uuid(value, label: str) -> UUID:
    try:
        return UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a UUID") from exc


def _object(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def load_cases(path: str | Path = DEFAULT_DATASET) -> tuple[GenerationCase, ...]:
    data = _object(json.loads(Path(path).read_text(encoding="utf-8")), "dataset")
    if set(data) != {"sources", "snapshots", "passages", "cases"}:
        raise ValueError("generation dataset requires sources, snapshots, passages, cases")
    sources = {}
    for key, raw in _object(data["sources"], "sources").items():
        row = _object(raw, f"source {key}")
        sources[key] = SourceRef(
            UUID(int=len(sources) + 1), _nonempty(row.get("source_key"), "source key"),
            _nonempty(row.get("title"), "source title"), _nonempty(row.get("url"), "source URL"),
        )
    snapshots = {}
    for key, raw in _object(data["snapshots"], "snapshots").items():
        row = _object(raw, f"snapshot {key}")
        if row.get("source") not in sources:
            raise ValueError(f"snapshot {key}: unknown source")
        snapshots[key] = SnapshotRef(
            _uuid(row.get("id"), f"snapshot {key} ID"), sources[row["source"]].id,
            _digest(row.get("hash"), f"snapshot {key} hash"), sources[row["source"]].canonical_url,
            _timestamp(row.get("observed_at"), f"snapshot {key} observed_at"), None, None, None,
        )
    passages = {}
    for key, raw in _object(data["passages"], "passages").items():
        row = _object(raw, f"passage {key}")
        if row.get("snapshot") not in snapshots:
            raise ValueError(f"passage {key}: unknown snapshot")
        ordinal = row.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError(f"passage {key}: ordinal must be nonnegative")
        snapshot = snapshots[row["snapshot"]]
        passages[key] = PassageRef(
            _uuid(row.get("id"), f"passage {key} ID"), snapshot.id, snapshot.observed_at,
            ordinal, None, _digest(row.get("hash"), f"passage {key} hash"),
            _nonempty(row.get("text"), f"passage {key} text"),
        )

    def evidence_for(key: str) -> Evidence:
        if key not in passages:
            raise ValueError(f"unknown passage reference: {key}")
        passage = passages[key]
        snapshot_key = data["passages"][key]["snapshot"]
        snapshot = snapshots[snapshot_key]
        source = sources[data["snapshots"][snapshot_key]["source"]]
        return Evidence(
            source.id, source.source_key, source.title, source.canonical_url,
            snapshot.id, snapshot.content_sha256, snapshot.retrieved_url,
            snapshot.observed_at, None, None, passage.id, passage.ordinal,
            passage.page, passage.text, passage.text_sha256, "generation_fixture", 0.0, 1,
            "not_ranked",
        )

    security = {case.case_id: case for case in load_security_cases()}
    cases = []
    seen = set()
    for raw in _list(data["cases"], "cases"):
        row = _object(raw, "case")
        case_id = _nonempty(row.get("id"), "case ID")
        if case_id in seen:
            raise ValueError(f"duplicate generation case ID: {case_id}")
        seen.add(case_id)
        status = row.get("expected_status")
        if status not in ("answered", "abstained", "rejected"):
            raise ValueError(f"{case_id}: invalid expected_status")
        category = row.get("expected_category")
        if (status == "rejected") != (isinstance(category, str) and bool(category)):
            raise ValueError(f"{case_id}: rejected cases require expected_category only")
        if "security_case_id" in row:
            if set(row) - {"id", "security_case_id", "expected_status", "expected_category"}:
                raise ValueError(f"{case_id}: security reference has extra fields")
            security_id = _nonempty(row["security_case_id"], f"{case_id} security_case_id")
            security_case = security.get(security_id)
            if security_case is None:
                raise ValueError(f"{case_id}: unknown security case")
            cases.append(GenerationCase(
                case_id, security_case.question, tuple(evidence_for_case(security_case)),
                security_case.provider_response, status, category,
            ))
            continue
        if set(row) - {"id", "question", "mode", "evidence", "provider_response", "expected_status",
                       "expected_category", "required_evidence", "requirement_mode", "expected_snapshots",
                       "as_of", "older_snapshot", "newer_snapshot", "expected_provider_calls"}:
            raise ValueError(f"{case_id}: unknown fields")
        question = _nonempty(row.get("question"), f"{case_id} question")
        if "provider_response" not in row:
            raise ValueError(f"{case_id}: provider_response is required")
        keys = _list(row.get("evidence"), f"{case_id} evidence")
        if any(not isinstance(key, str) or not key.strip() for key in keys):
            raise ValueError(f"{case_id}: evidence references must be non-empty strings")
        if len(set(keys)) != len(keys):
            raise ValueError(f"{case_id}: duplicate evidence reference")
        supplied = tuple(evidence_for(key) for key in keys)
        mode = row.get("mode")
        if mode not in ("answer", "change"):
            raise ValueError(f"{case_id}: invalid mode")
        changeset = None
        if mode == "change":
            old_key, new_key = row.get("older_snapshot"), row.get("newer_snapshot")
            if old_key not in snapshots or new_key not in snapshots:
                raise ValueError(f"{case_id}: unknown change snapshot")
            older, newer = snapshots[old_key], snapshots[new_key]
            source = next((item for item in sources.values() if item.id == older.source_id), None)
            if source is None:
                raise ValueError(f"{case_id}: unknown change source")
            if any(item.snapshot_id not in (older.id, newer.id) for item in supplied):
                raise ValueError(f"{case_id}: change evidence must belong to compared snapshots")
            changeset = compare_passages(
                source, older, newer,
                [passages[key] for key in keys if passages[key].snapshot_id == older.id],
                [passages[key] for key in keys if passages[key].snapshot_id == newer.id],
            )
        elif "older_snapshot" in row or "newer_snapshot" in row:
            raise ValueError(f"{case_id}: snapshot pair requires change mode")
        requirement_mode = row.get("requirement_mode", "all")
        if requirement_mode not in ("all", "any"):
            raise ValueError(f"{case_id}: requirement_mode must be all or any")
        requirements = []
        for wanted in _list(row.get("required_evidence", []), f"{case_id} required_evidence"):
            if isinstance(wanted, str):
                item = evidence_for(wanted)
                requirements.append(EvidenceRequirement(
                    item.source_key, item.snapshot_id, item.snapshot_content_sha256,
                    item.passage_id, item.text_sha256,
                ))
            else:
                spec = _object(wanted, f"{case_id} evidence requirement")
                if not spec or set(spec) - {"source_key", "snapshot_id", "snapshot_hash", "passage_id", "passage_hash"}:
                    raise ValueError(f"{case_id}: invalid evidence requirement")
                if "source_key" in spec:
                    _nonempty(spec["source_key"], f"{case_id} required source_key")
                requirements.append(EvidenceRequirement(
                    source_key=spec.get("source_key"),
                    snapshot_id=_uuid(spec["snapshot_id"], "required snapshot ID") if "snapshot_id" in spec else None,
                    snapshot_hash=_digest(spec["snapshot_hash"], "required snapshot hash") if "snapshot_hash" in spec else None,
                    passage_id=_uuid(spec["passage_id"], "required passage ID") if "passage_id" in spec else None,
                    passage_hash=_digest(spec["passage_hash"], "required passage hash") if "passage_hash" in spec else None,
                ))
        snapshot_ids = []
        for key in _list(row.get("expected_snapshots", []), f"{case_id} expected_snapshots"):
            if not isinstance(key, str):
                raise ValueError(f"{case_id}: expected snapshot references must be strings")
            if key not in snapshots:
                raise ValueError(f"{case_id}: unknown expected snapshot")
            snapshot_ids.append(snapshots[key].id)
        if len(set(snapshot_ids)) != len(snapshot_ids):
            raise ValueError(f"{case_id}: duplicate expected snapshot")
        as_of = _timestamp(row["as_of"], f"{case_id} as_of") if "as_of" in row else None
        if status != "answered" and (requirements or snapshot_ids):
            raise ValueError(f"{case_id}: evidence expectations require answered status")
        if as_of is not None and not snapshot_ids:
            raise ValueError(f"{case_id}: as_of requires expected_snapshots")
        calls = row.get("expected_provider_calls", 0 if not supplied else 1)
        if type(calls) is not int or calls not in (0, 1):
            raise ValueError(f"{case_id}: expected_provider_calls must be 0 or 1")
        cases.append(GenerationCase(
            case_id, question, supplied, row.get("provider_response"), status, category,
            tuple(requirements), requirement_mode, tuple(snapshot_ids), as_of, changeset, calls,
        ))
    if not cases:
        raise ValueError("generation dataset must contain cases")
    return tuple(cases)


class _FakeProvider:
    def __init__(self, response: object):
        self.response = response
        self.calls = 0

    def generate(self, question: str, formatted_evidence: str, instructions: str) -> object:
        self.calls += 1
        return self.response


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_cases(cases: tuple[GenerationCase, ...]) -> GenerationEvaluationReport:
    observations = []
    status_correct = answerability_total = answerability_correct = 0
    correct_abstentions = incorrect_abstentions = incorrect_answers = 0
    claims = covered_claims = citations = valid_citations = 0
    expected_total = expected_correct = temporal_total = temporal_correct = 0
    rejection_categories: dict[str, int] = {}
    diagnostic_counts: dict[str, int] = {}
    for case in cases:
        provider = _FakeProvider(case.provider_response)
        service = EvidenceFirstAnswerService(provider)
        answer = None
        category = None
        try:
            answer = (VersionChangeSummaryService(service).summarize(case.change_set, case.question)
                      if case.change_set is not None else service.answer(case.question, case.evidence))
            observed: Status = answer.status
        except AnswerError as error:
            observed = "rejected"
            category = failure_category(error)
            rejection_categories[category] = rejection_categories.get(category, 0) + 1
        status_correct += int(observed == case.expected_status)
        if case.expected_status in ("answered", "abstained"):
            answerability_total += 1
            answerability_correct += int(observed == case.expected_status)
            correct_abstentions += int(case.expected_status == "abstained" and observed == "abstained")
            incorrect_abstentions += int(case.expected_status == "answered" and observed == "abstained")
            incorrect_answers += int(case.expected_status == "abstained" and observed == "answered")
        cited = [citation.evidence for claim in answer.claims for citation in claim.citations] if answer and observed == "answered" else []
        if observed == "answered" and answer:
            claims += len(answer.claims)
            covered_claims += sum(bool(claim.citations) for claim in answer.claims)
            citations += len(cited)
            valid_citations += len(cited)  # Only citations returned by the real validator are counted.
        evidence_match = None
        if case.requirements:
            expected_total += 1
            matches = [any(req.matches(item) for item in cited) for req in case.requirements]
            evidence_match = observed == "answered" and (all(matches) if case.requirement_mode == "all" else any(matches))
            expected_correct += int(evidence_match)
        temporal_match = None
        if case.expected_snapshot_ids:
            temporal_total += 1
            temporal_match = (
                observed == "answered"
                and {item.snapshot_id for item in cited} == set(case.expected_snapshot_ids)
                and (case.as_of is None or all(item.observed_at <= case.as_of for item in cited))
            )
            temporal_correct += int(temporal_match)
        diagnostics = []
        if observed != case.expected_status:
            diagnostics.append("unexpected_rejection" if observed == "rejected" else "generation_status_failure")
        elif observed == "rejected":
            diagnostics.append("expected_rejection" if category == case.expected_category else "citation_validation_failure")
        if case.expected_provider_calls != provider.calls:
            diagnostics.append("provider_call_failure")
        if observed == "answered":
            if temporal_match is False:
                diagnostics.append("temporal_version_failure")
            elif evidence_match is False:
                diagnostics.append("evidence_selection_failure")
        passed = (observed == case.expected_status and category == case.expected_category
                  and provider.calls == case.expected_provider_calls
                  and evidence_match is not False and temporal_match is not False)
        for diagnostic in diagnostics:
            diagnostic_counts[diagnostic] = diagnostic_counts.get(diagnostic, 0) + 1
        observations.append(CaseObservation(case.case_id, observed, category, provider.calls,
                                            passed, tuple(diagnostics), evidence_match, temporal_match))
    failed = tuple(item.case_id for item in observations if not item.passed)
    return GenerationEvaluationReport(
        cases=len(cases), passed=len(cases) - len(failed), failed=len(failed), failed_case_ids=failed,
        answer_status_accuracy=_ratio(status_correct, len(cases)),
        abstention_accuracy=_ratio(answerability_correct, answerability_total),
        correct_abstentions=correct_abstentions, incorrect_abstentions=incorrect_abstentions,
        incorrect_answers=incorrect_answers, citation_coverage=_ratio(covered_claims, claims),
        citation_validity=_ratio(valid_citations, citations),
        expected_evidence_accuracy=_ratio(expected_correct, expected_total),
        temporal_evidence_accuracy=_ratio(temporal_correct, temporal_total),
        contract_pass_rate=_ratio(len(cases) - len(failed), len(cases)),
        rejection_categories=rejection_categories, diagnostic_counts=diagnostic_counts,
        observations=tuple(observations),
    )
