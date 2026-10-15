import copy
import hashlib
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID

from sqlalchemy.dialects import postgresql

from api.content_hash import sha256_bytes, sha256_text
from api.versioned_ingest import ObservedSourceVersion, VersionedIngestionService
from api.versioned_parser import ParsedPassage, parse_pdf_passages
from api.versioned_store import VersionedCorpusStore


class MemorySessionFactory:
    """Transactional fake for service tests; it does not emulate PostgreSQL constraints."""

    def __init__(self):
        self.state = SimpleNamespace(sources={}, snapshots=[], passages=[], next_id=1)
        self.commits = 0
        self.rollbacks = 0
        self.fail_commit = False

    @contextmanager
    def begin(self):
        working = copy.deepcopy(self.state)
        try:
            yield working
            if self.fail_commit:
                raise RuntimeError("commit failed")
        except Exception:
            self.rollbacks += 1
            raise
        else:
            self.state = working
            self.commits += 1


class MemoryStore:
    def __init__(self, state, *, fail_passages=False):
        self.state = state
        self.fail_passages = fail_passages

    def _id(self):
        identifier = UUID(int=self.state.next_id)
        self.state.next_id += 1
        return identifier

    def get_source(self, source_key):
        return self.state.sources.get(source_key)

    def add_source(self, source_key, canonical_url, title):
        source = SimpleNamespace(id=self._id(), source_key=source_key, canonical_url=canonical_url, title=title)
        self.state.sources[source_key] = source
        return source

    def update_source_metadata(self, source, canonical_url, title):
        source.canonical_url = canonical_url
        source.title = title

    def get_snapshot_by_hash(self, source_id, content_sha256):
        return next(
            (snapshot for snapshot in self.state.snapshots
             if snapshot.source_id == source_id and snapshot.content_sha256 == content_sha256),
            None,
        )

    def get_latest_snapshot(self, source_id):
        snapshots = [snapshot for snapshot in self.state.snapshots if snapshot.source_id == source_id]
        return max(snapshots, key=lambda snapshot: (snapshot.observed_at, snapshot.id.int), default=None)

    def add_snapshot(self, **fields):
        snapshot = SimpleNamespace(id=self._id(), **fields)
        self.state.snapshots.append(snapshot)
        return snapshot

    def add_passages(self, snapshot_id, passages):
        for ordinal, passage in enumerate(passages):
            self.state.passages.append(
                SimpleNamespace(
                    id=self._id(), snapshot_id=snapshot_id, ordinal=ordinal,
                    page=passage.page, text=passage.text, text_sha256=sha256_text(passage.text),
                )
            )
            if self.fail_passages:
                raise RuntimeError("passage persistence failed")

    def passage_count(self, snapshot_id):
        return sum(passage.snapshot_id == snapshot_id for passage in self.state.passages)


class VersionedIngestTests(unittest.TestCase):
    def setUp(self):
        self.sessions = MemorySessionFactory()
        self.parser = Mock(side_effect=lambda content: [
            ParsedPassage(page=1, text=content.decode("utf-8")),
            ParsedPassage(page=2, text="Second passage"),
        ])
        self.fail_passages = False
        self.service = VersionedIngestionService(
            self.sessions,
            parser=self.parser,
            store_factory=lambda session: MemoryStore(session, fail_passages=self.fail_passages),
        )

    def observed(self, content=b"Version one", source_key="source-a", **changes):
        values = dict(
            source_key=source_key,
            canonical_url="https://example.test/source-a",
            title="Example source",
            retrieved_url="https://example.test/source-a.pdf",
            content=content,
            observed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        values.update(changes)
        return ObservedSourceVersion(**values)

    def test_first_ingestion_creates_source_snapshot_and_hashed_passages(self):
        result = self.service.ingest(self.observed())
        self.assertEqual(result.status, "new_source")
        self.assertTrue(result.content_changed)
        self.assertEqual(result.content_sha256, hashlib.sha256(b"Version one").hexdigest())
        self.assertEqual(len(self.sessions.state.sources), 1)
        self.assertEqual(len(self.sessions.state.snapshots), 1)
        self.assertIsNone(self.sessions.state.snapshots[0].predecessor_snapshot_id)
        self.assertEqual(result.passage_count, 2)
        self.assertEqual([passage.ordinal for passage in self.sessions.state.passages], [0, 1])
        self.assertEqual([passage.page for passage in self.sessions.state.passages], [1, 2])
        self.assertTrue(all(p.snapshot_id == result.snapshot_id for p in self.sessions.state.passages))
        self.assertEqual(self.sessions.state.passages[0].text_sha256, sha256_text("Version one"))

    def test_exact_repeat_is_a_persisted_no_op(self):
        first = self.service.ingest(self.observed())
        original_state = copy.deepcopy(self.sessions.state)
        repeat = self.service.ingest(self.observed(title="Changed title"))
        self.assertEqual(repeat.status, "unchanged")
        self.assertFalse(repeat.content_changed)
        self.assertEqual(repeat.source_id, first.source_id)
        self.assertEqual(repeat.snapshot_id, first.snapshot_id)
        self.assertEqual(repeat.passage_count, 2)
        self.assertEqual(self.sessions.state.sources["source-a"].title, "Example source")
        self.assertEqual(vars(self.sessions.state), vars(original_state))
        self.assertEqual(self.parser.call_count, 1)

    def test_changed_content_preserves_history_and_links_predecessor(self):
        first = self.service.ingest(self.observed())
        old_passages = copy.deepcopy(self.sessions.state.passages)
        changed = self.service.ingest(self.observed(
            content=b"Version two", title="Updated source",
            observed_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
        ))
        self.assertEqual(changed.status, "new_snapshot")
        self.assertEqual(changed.source_id, first.source_id)
        self.assertNotEqual(changed.snapshot_id, first.snapshot_id)
        self.assertEqual(len(self.sessions.state.sources), 1)
        self.assertEqual(len(self.sessions.state.snapshots), 2)
        self.assertEqual(self.sessions.state.snapshots[1].predecessor_snapshot_id, first.snapshot_id)
        self.assertEqual(self.sessions.state.passages[:2], old_passages)
        self.assertTrue(all(p.snapshot_id == changed.snapshot_id for p in self.sessions.state.passages[2:]))
        self.assertEqual(self.sessions.state.sources["source-a"].title, "Updated source")

    def test_same_content_is_allowed_for_independent_sources(self):
        first = self.service.ingest(self.observed())
        second = self.service.ingest(self.observed(source_key="source-b"))
        self.assertEqual(second.status, "new_source")
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertNotEqual(first.source_id, second.source_id)
        self.assertEqual(len(self.sessions.state.snapshots), 2)

    def test_supplied_source_timestamps_are_persisted(self):
        published = datetime(2024, 12, 1, tzinfo=timezone.utc)
        updated = datetime(2024, 12, 15, tzinfo=timezone.utc)
        effective = datetime(2025, 1, 2, tzinfo=timezone.utc)
        observation = self.observed(
            source_published_at=published, source_updated_at=updated, effective_from=effective,
        )
        self.service.ingest(observation)
        snapshot = self.sessions.state.snapshots[0]
        self.assertEqual(snapshot.observed_at, observation.observed_at)
        self.assertEqual(snapshot.source_published_at, published)
        self.assertEqual(snapshot.source_updated_at, updated)
        self.assertEqual(snapshot.effective_from, effective)
        self.assertEqual(snapshot.retrieved_url, observation.retrieved_url)

    def test_failure_after_snapshot_and_passage_work_rolls_back(self):
        first = self.service.ingest(self.observed())
        old_state = copy.deepcopy(self.sessions.state)
        self.fail_passages = True
        with self.assertRaisesRegex(RuntimeError, "passage persistence failed"):
            self.service.ingest(self.observed(content=b"Version two", title="Uncommitted title"))
        self.assertEqual(vars(self.sessions.state), vars(old_state))
        self.assertEqual(len(self.sessions.state.snapshots), 1)
        self.assertEqual(self.sessions.state.snapshots[0].id, first.snapshot_id)
        self.assertEqual(self.sessions.rollbacks, 1)

    def test_commit_failure_and_parser_failure_leave_no_new_version(self):
        self.service.ingest(self.observed())
        old_state = copy.deepcopy(self.sessions.state)
        self.sessions.fail_commit = True
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            self.service.ingest(self.observed(content=b"Version two"))
        self.sessions.fail_commit = False
        self.parser.side_effect = ValueError("cannot parse")
        with self.assertRaisesRegex(ValueError, "cannot parse"):
            self.service.ingest(self.observed(content=b"Version three"))
        self.assertEqual(vars(self.sessions.state), vars(old_state))
        self.assertEqual(self.sessions.rollbacks, 2)

    def test_input_requires_timezone_aware_timestamps(self):
        with self.assertRaisesRegex(ValueError, "observed_at"):
            self.observed(observed_at=datetime(2025, 1, 1))

    def test_hashing_uses_exact_bytes_and_utf8_text(self):
        self.assertEqual(sha256_bytes(b"a\r\nb"), hashlib.sha256(b"a\r\nb").hexdigest())
        self.assertEqual(sha256_text("caf\u00e9"), hashlib.sha256("caf\u00e9".encode("utf-8")).hexdigest())

    def test_pdf_parser_preserves_page_order_without_network(self):
        paragraphs = [
            {"page": 2, "text": "Second page paragraph."},
            {"page": 1, "text": "First page paragraph."},
        ]
        with patch("api.versioned_parser.load_pdf_from_bytes", return_value={"paragraphs": paragraphs}):
            with patch("api.chunker._get_encoder", side_effect=AssertionError("tokenizer must remain local")):
                passages = parse_pdf_passages(b"local fixture bytes")
        self.assertEqual([passage.page for passage in passages], [1, 2])


class VersionedStoreTests(unittest.TestCase):
    def test_sqlalchemy_passages_use_snapshot_ordinals_and_exact_text_hashes(self):
        session = Mock()
        snapshot_id = UUID(int=7)
        VersionedCorpusStore(session).add_passages(
            snapshot_id,
            [ParsedPassage(page=3, text="First"), ParsedPassage(page=None, text="Second")],
        )
        rows = session.add_all.call_args.args[0]
        self.assertEqual([row.snapshot_id for row in rows], [snapshot_id, snapshot_id])
        self.assertEqual([row.ordinal for row in rows], [0, 1])
        self.assertEqual([row.page for row in rows], [3, None])
        self.assertEqual([row.text_sha256 for row in rows], [sha256_text("First"), sha256_text("Second")])
        session.flush.assert_called_once()

    def test_latest_snapshot_query_has_deterministic_tie_breaks(self):
        session = Mock()
        VersionedCorpusStore(session).get_latest_snapshot(UUID(int=7))
        statement = session.scalar.call_args.args[0]
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertIn(
            "ORDER BY source_snapshots.observed_at DESC, "
            "source_snapshots.created_at DESC, source_snapshots.id DESC",
            sql,
        )


if __name__ == "__main__":
    unittest.main()
