"""Offline version comparison and evidence-bound change-summary tests."""

import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

from api.evidence_answer import (
    EvidenceFirstAnswerService, InvalidAnswerStructureError, UnknownEvidenceIdError,
)
from api.version_change import VersionChangeSummaryService, VersionComparisonError, change_evidence
from api.version_comparison import VersionComparisonService


T1 = datetime(2026, 10, 15, tzinfo=timezone.utc)
T2 = datetime(2026, 10, 16, tzinfo=timezone.utc)
SOURCE = UUID(int=1)
OTHER = UUID(int=2)
OLD = UUID(int=3)
NEW = UUID(int=4)


def passage(snapshot, ordinal, digest, text):
    return SimpleNamespace(id=UUID(int=100 + snapshot.int * 10 + ordinal), snapshot_id=snapshot,
                           ordinal=ordinal, page=None, text_sha256=digest, text=text)


class MemoryStore:
    def __init__(self, snapshots, passages):
        self.snapshots = {snapshot.id: snapshot for snapshot in snapshots}
        self.passages = passages
        self.sources = {
            SOURCE: SimpleNamespace(id=SOURCE, source_key="govuk:fixture", title="Illustrative guidance",
                                    canonical_url="https://www.gov.uk/example"),
            OTHER: SimpleNamespace(id=OTHER, source_key="govuk:other", title="Other",
                                   canonical_url="https://www.gov.uk/other"),
        }

    def get_snapshot(self, snapshot_id):
        return self.snapshots.get(snapshot_id)

    def get_source_by_id(self, source_id):
        return self.sources.get(source_id)

    def get_passages(self, snapshot_id):
        return sorted(self.passages[snapshot_id], key=lambda item: item.ordinal)

    def get_latest_snapshot(self, source_id, as_of=None):
        matches = [snapshot for snapshot in self.snapshots.values()
                   if snapshot.source_id == source_id and (as_of is None or snapshot.observed_at <= as_of)]
        return max(matches, key=lambda item: (item.observed_at, item.created_at, item.id), default=None)


class Session:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Provider:
    def __init__(self, output=None):
        self.output = output
        self.calls = []

    def generate(self, question, formatted_evidence, instructions):
        self.calls.append((question, formatted_evidence, instructions))
        return self.output


class VersionChangeTests(unittest.TestCase):
    def setUp(self):
        self.old = SimpleNamespace(id=OLD, source_id=SOURCE, content_sha256="a" * 64,
                                   retrieved_url="https://www.gov.uk/api/content/example", observed_at=T1,
                                   source_published_at=None, source_updated_at=None,
                                   predecessor_snapshot_id=None, created_at=T1)
        self.new = SimpleNamespace(id=NEW, source_id=SOURCE, content_sha256="b" * 64,
                                   retrieved_url="https://www.gov.uk/api/content/example", observed_at=T2,
                                   source_published_at=None, source_updated_at=None,
                                   predecessor_snapshot_id=OLD, created_at=T2)
        self.store = MemoryStore([self.old, self.new], {
            OLD: [passage(OLD, 0, "a", "Previous wording"), passage(OLD, 1, "b", "Stable wording")],
            NEW: [passage(NEW, 0, "c", "Newer wording"), passage(NEW, 1, "b", "Stable wording")],
        })
        self.service = VersionComparisonService(lambda: Session(), store_factory=lambda _: self.store)

    def test_modified_and_unchanged_with_provenance(self):
        result = self.service.compare_snapshots(OLD, NEW)
        self.assertEqual([change.kind for change in result.changes], ["modified", "unchanged"])
        self.assertEqual((result.counts.modified, result.counts.unchanged), (1, 1))
        self.assertEqual((result.counts.total_older, result.counts.total_newer), (2, 2))
        self.assertEqual(result.source.id, SOURCE)
        self.assertEqual(result.older.content_sha256, "a" * 64)
        self.assertEqual(result.newer.observed_at, T2)
        self.assertEqual(result.changes[0].older.id, self.store.passages[OLD][0].id)
        self.assertEqual(result.changes[0].newer.text_sha256, "c")
        self.assertEqual(result.changes[0].older.observed_at, T1)

    def test_zero_change_short_circuits_provider(self):
        self.store.passages[NEW][0].text_sha256 = "a"
        self.store.passages[NEW][0].text = "Different text with the same stored hash"
        changes = self.service.compare_snapshots(OLD, NEW)
        self.assertEqual([change.kind for change in changes.changes], ["unchanged", "unchanged"])
        self.assertEqual((changes.counts.added, changes.counts.removed, changes.counts.modified), (0, 0, 0))
        provider = Provider()
        summary = VersionChangeSummaryService(EvidenceFirstAnswerService(provider)).summarize(changes)
        self.assertEqual(summary.status, "abstained")
        self.assertEqual(summary.abstention_reason, "No observed content changes.")
        self.assertEqual(provider.calls, [])

    def test_added_removed_and_mixed_order_and_counts(self):
        self.store.passages[OLD] = [passage(OLD, 0, "a", "old"), passage(OLD, 1, "b", "stable"),
                                    passage(OLD, 3, "d", "removed")]
        self.store.passages[NEW] = [passage(NEW, 0, "c", "new"), passage(NEW, 1, "b", "stable"),
                                    passage(NEW, 2, "e", "added")]
        result = self.service.compare_snapshots(OLD, NEW)
        self.assertEqual([(c.ordinal, c.kind) for c in result.changes],
                         [(0, "modified"), (1, "unchanged"), (2, "added"), (3, "removed")])
        self.assertEqual((result.counts.unchanged, result.counts.added, result.counts.removed,
                          result.counts.modified, result.counts.total_older, result.counts.total_newer),
                         (1, 1, 1, 1, 3, 3))
        self.assertIsNone(result.changes[2].older)
        self.assertIsNone(result.changes[3].newer)

    def test_all_change_counts(self):
        self.store.passages[OLD] = [passage(OLD, 0, "a", "removed")]
        self.store.passages[NEW] = [passage(NEW, 1, "b", "added")]
        result = self.service.compare_snapshots(OLD, NEW)
        self.assertEqual((result.counts.unchanged, result.counts.added, result.counts.removed,
                          result.counts.modified, result.counts.total_older, result.counts.total_newer),
                         (0, 1, 1, 0, 1, 1))

    def test_rejects_different_sources_self_missing_and_reversed(self):
        for args, message in [((OLD, OLD), "itself"), ((OLD, UUID(int=99)), "not found"),
                              ((NEW, OLD), "earlier")]:
            with self.subTest(args=args), self.assertRaisesRegex(VersionComparisonError, message):
                self.service.compare_snapshots(*args)
        self.new.source_id = OTHER
        with self.assertRaisesRegex(VersionComparisonError, "same source"):
            self.service.compare_snapshots(OLD, NEW)
        with self.assertRaisesRegex(VersionComparisonError, "UUID"):
            self.service.compare_snapshots("bad", NEW)

    def test_predecessor_and_as_of_resolution(self):
        self.assertEqual(self.service.compare_predecessor(NEW).older.id, OLD)
        with self.assertRaisesRegex(VersionComparisonError, "no predecessor"):
            self.service.compare_predecessor(OLD)
        self.assertEqual(self.service.compare_as_of(SOURCE, T1, T2).newer.id, NEW)
        with self.assertRaisesRegex(VersionComparisonError, "itself"):
            self.service.compare_as_of(SOURCE, T1, T1)
        with self.assertRaisesRegex(VersionComparisonError, "cutoff"):
            self.service.compare_as_of(SOURCE, T2, T1)
        with self.assertRaisesRegex(VersionComparisonError, "timezone-aware"):
            self.service.compare_as_of(SOURCE, T1.replace(tzinfo=None), T2)

    def test_summary_labels_both_observations_and_validates_quotes(self):
        changes = self.service.compare_snapshots(OLD, NEW)
        output = {"status": "answered", "claims": [{"text": "The wording changed.", "citations": [
            {"evidence_id": "E1", "supporting_quote": "Previous wording"},
            {"evidence_id": "E2", "supporting_quote": "Newer wording"},
        ]}], "abstention_reason": None}
        provider = Provider(output)
        answer = VersionChangeSummaryService(EvidenceFirstAnswerService(provider)).summarize(changes)
        self.assertEqual(answer.status, "answered")
        self.assertEqual([c.evidence.snapshot_id for c in answer.claims[0].citations], [OLD, NEW])
        blocks = [json.loads(line) for line in provider.calls[0][1].splitlines()]
        self.assertEqual([block["version_change"]["observation_side"] for block in blocks], ["older", "newer"])
        self.assertEqual([block["version_change"]["change_kind"] for block in blocks], ["modified", "modified"])
        self.assertIn("previous observed wording", provider.calls[0][2].lower())
        self.assertEqual(len(change_evidence(changes)[0]), 2)

    def test_summary_rejects_fabricated_and_one_sided_modified_claim(self):
        changes = self.service.compare_snapshots(OLD, NEW)
        for evidence_id, error in [("E99", UnknownEvidenceIdError),
                                   ("E1", InvalidAnswerStructureError)]:
            provider = Provider({"status": "answered", "claims": [{"text": "Changed.", "citations": [
                {"evidence_id": evidence_id, "supporting_quote": "Previous wording"}
            ]}], "abstention_reason": None})
            with self.subTest(evidence_id=evidence_id), self.assertRaises(error):
                VersionChangeSummaryService(EvidenceFirstAnswerService(provider)).summarize(changes)

    def test_added_and_removed_claims_can_cite_their_own_side(self):
        self.store.passages[OLD] = [passage(OLD, 0, "a", "Removed wording")]
        self.store.passages[NEW] = [passage(NEW, 1, "b", "Added wording")]
        changes = self.service.compare_snapshots(OLD, NEW)
        provider = Provider({"status": "answered", "claims": [
            {"text": "The previous observation contained this wording.", "citations": [
                {"evidence_id": "E1", "supporting_quote": "Removed wording"}]},
            {"text": "The newer observation contains this wording.", "citations": [
                {"evidence_id": "E2", "supporting_quote": "Added wording"}]},
        ], "abstention_reason": None})
        result = VersionChangeSummaryService(EvidenceFirstAnswerService(provider)).summarize(changes)
        self.assertEqual(result.status, "answered")
        self.assertEqual([c.evidence.snapshot_id for claim in result.claims for c in claim.citations],
                         [OLD, NEW])


if __name__ == "__main__":
    unittest.main()
