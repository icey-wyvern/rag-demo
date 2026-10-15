"""Derived, ordinal-based comparison of two immutable source observations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from .evidence import Evidence
from .evidence_answer import (
    AnswerResult, EvidenceFirstAnswerService, InvalidAnswerStructureError,
)


class VersionComparisonError(ValueError):
    """The requested pair cannot be compared."""


@dataclass(frozen=True)
class SourceRef:
    id: UUID
    source_key: str
    title: str
    canonical_url: str


@dataclass(frozen=True)
class SnapshotRef:
    id: UUID
    source_id: UUID
    content_sha256: str
    retrieved_url: str
    observed_at: datetime
    source_published_at: datetime | None
    source_updated_at: datetime | None
    predecessor_snapshot_id: UUID | None


@dataclass(frozen=True)
class PassageRef:
    id: UUID
    snapshot_id: UUID
    observed_at: datetime
    ordinal: int
    page: int | None
    text_sha256: str
    text: str


ChangeKind = Literal["unchanged", "added", "removed", "modified"]


@dataclass(frozen=True)
class PassageChange:
    kind: ChangeKind
    ordinal: int
    older: PassageRef | None
    newer: PassageRef | None


@dataclass(frozen=True)
class ChangeCounts:
    unchanged: int
    added: int
    removed: int
    modified: int
    total_older: int
    total_newer: int


@dataclass(frozen=True)
class VersionChangeSet:
    source: SourceRef
    older: SnapshotRef
    newer: SnapshotRef
    changes: tuple[PassageChange, ...]
    counts: ChangeCounts

    @property
    def material_changes(self) -> tuple[PassageChange, ...]:
        return tuple(change for change in self.changes if change.kind != "unchanged")


def compare_passages(
    source: SourceRef, older: SnapshotRef, newer: SnapshotRef,
    older_passages: list[PassageRef], newer_passages: list[PassageRef],
) -> VersionChangeSet:
    if older.id == newer.id:
        raise VersionComparisonError("Cannot compare a snapshot to itself")
    if older.source_id != newer.source_id or source.id != older.source_id:
        raise VersionComparisonError("Snapshots must belong to the same source")
    if older.observed_at >= newer.observed_at:
        raise VersionComparisonError("Older snapshot must have an earlier observation time")
    old_by_ordinal = {passage.ordinal: passage for passage in older_passages}
    new_by_ordinal = {passage.ordinal: passage for passage in newer_passages}
    if len(old_by_ordinal) != len(older_passages) or len(new_by_ordinal) != len(newer_passages):
        raise VersionComparisonError("Duplicate passage ordinal")
    changes = []
    counts = {kind: 0 for kind in ("unchanged", "added", "removed", "modified")}
    for ordinal in sorted(old_by_ordinal.keys() | new_by_ordinal.keys()):
        old, new = old_by_ordinal.get(ordinal), new_by_ordinal.get(ordinal)
        kind: ChangeKind = (
            "added" if old is None else "removed" if new is None else
            "unchanged" if old.text_sha256 == new.text_sha256 else "modified"
        )
        changes.append(PassageChange(kind, ordinal, old, new))
        counts[kind] += 1
    summary = ChangeCounts(**counts, total_older=len(older_passages), total_newer=len(newer_passages))
    return VersionChangeSet(source, older, newer, tuple(changes), summary)


def change_evidence(changeset: VersionChangeSet) -> tuple[list[Evidence], dict[str, dict[str, str | int]], dict[int, tuple[str, str]]]:
    """Old and new passages remain separate, labeled prompt-local evidence."""
    evidence = []
    context: dict[str, dict[str, str | int]] = {}
    modified_pairs: dict[int, tuple[str, str]] = {}
    for change in changeset.material_changes:
        ids = {}
        for side, passage, snapshot in (
            ("older", change.older, changeset.older),
            ("newer", change.newer, changeset.newer),
        ):
            if passage is None or not passage.text.strip():
                continue
            evidence_id = f"E{len(evidence) + 1}"
            ids[side] = evidence_id
            evidence.append(Evidence(
                source_id=changeset.source.id, source_key=changeset.source.source_key,
                source_title=changeset.source.title, canonical_url=changeset.source.canonical_url,
                snapshot_id=snapshot.id, snapshot_content_sha256=snapshot.content_sha256,
                retrieved_url=snapshot.retrieved_url, observed_at=snapshot.observed_at,
                source_published_at=snapshot.source_published_at,
                source_updated_at=snapshot.source_updated_at, passage_id=passage.id,
                passage_ordinal=passage.ordinal, page=passage.page, text=passage.text,
                text_sha256=passage.text_sha256, retrieval_method="version_change",
                score=0.0, rank=len(evidence), score_kind="not_ranked",
            ))
            context[evidence_id] = {
                "observation_side": side, "change_kind": change.kind,
                "passage_ordinal": change.ordinal,
            }
        if change.kind == "modified" and "older" in ids and "newer" in ids:
            modified_pairs[change.ordinal] = (ids["older"], ids["newer"])
    return evidence, context, modified_pairs


class VersionChangeSummaryService:
    def __init__(self, answer_service: EvidenceFirstAnswerService):
        self.answer_service = answer_service

    def summarize(self, changeset: VersionChangeSet, question: str = "What changed between these observations?") -> AnswerResult:
        if not changeset.material_changes:
            return AnswerResult("abstained", (), "No observed content changes.")
        evidence, context, pairs = change_evidence(changeset)
        instructions = (
            "Describe observed wording/content changes, additions and removals. "
            "Older evidence is previous observed wording, not current guidance; newer evidence is newer observed wording. "
            "A modified passage claim must cite both observations. Different text alone does not prove a semantic contradiction."
        )
        result = self.answer_service.answer(question, evidence, evidence_context=context, additional_instructions=instructions)
        for claim in result.claims:
            cited = {citation.evidence_id for citation in claim.citations}
            for old_id, new_id in pairs.values():
                if (old_id in cited) != (new_id in cited):
                    raise InvalidAnswerStructureError("Modified passage claims require older and newer citations")
        return result
