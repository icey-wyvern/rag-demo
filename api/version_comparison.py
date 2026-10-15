"""Read-only snapshot resolution and comparison for one logical source."""

from datetime import datetime
from uuid import UUID

from .version_change import (
    PassageRef, SnapshotRef, SourceRef, VersionChangeSet,
    VersionComparisonError, compare_passages,
)
from .versioned_store import VersionedCorpusStore


def _uuid(value: UUID) -> UUID:
    if not isinstance(value, UUID):
        raise VersionComparisonError("Snapshot and source IDs must be UUID values")
    return value


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise VersionComparisonError("Observation cutoff must be timezone-aware")
    return value


class VersionComparisonService:
    def __init__(self, session_factory, store_factory=VersionedCorpusStore):
        self.session_factory = session_factory
        self.store_factory = store_factory

    def compare_snapshots(self, older_id: UUID, newer_id: UUID) -> VersionChangeSet:
        _uuid(older_id)
        _uuid(newer_id)
        with self.session_factory() as session:
            store = self.store_factory(session)
            return self._compare(store, older_id, newer_id)

    def compare_predecessor(self, snapshot_id: UUID) -> VersionChangeSet:
        _uuid(snapshot_id)
        with self.session_factory() as session:
            store = self.store_factory(session)
            snapshot = store.get_snapshot(snapshot_id)
            if snapshot is None:
                raise VersionComparisonError("Snapshot not found")
            if snapshot.predecessor_snapshot_id is None:
                raise VersionComparisonError("Snapshot has no predecessor")
            return self._compare(store, snapshot.predecessor_snapshot_id, snapshot_id)

    def compare_as_of(self, source_id: UUID, older_as_of: datetime, newer_as_of: datetime) -> VersionChangeSet:
        _uuid(source_id)
        _aware(older_as_of)
        _aware(newer_as_of)
        if older_as_of > newer_as_of:
            raise VersionComparisonError("Older cutoff must not be after newer cutoff")
        with self.session_factory() as session:
            store = self.store_factory(session)
            older = store.get_latest_snapshot(source_id, older_as_of)
            newer = store.get_latest_snapshot(source_id, newer_as_of)
            if older is None or newer is None:
                raise VersionComparisonError("No snapshot observed by one or both cutoffs")
            return self._compare(store, older.id, newer.id)

    @staticmethod
    def _compare(store, older_id: UUID, newer_id: UUID) -> VersionChangeSet:
        if older_id == newer_id:
            raise VersionComparisonError("Cannot compare a snapshot to itself")
        older = store.get_snapshot(older_id)
        newer = store.get_snapshot(newer_id)
        if older is None or newer is None:
            raise VersionComparisonError("Snapshot not found")
        if older.source_id != newer.source_id:
            raise VersionComparisonError("Snapshots must belong to the same source")
        if older.observed_at >= newer.observed_at:
            raise VersionComparisonError("Older snapshot must have an earlier observation time")
        source = store.get_source_by_id(older.source_id)
        if source is None:
            raise VersionComparisonError("Source not found")
        source_ref = SourceRef(source.id, source.source_key, source.title, source.canonical_url)
        def snapshot_ref(snapshot):
            return SnapshotRef(
                snapshot.id, snapshot.source_id, snapshot.content_sha256,
                snapshot.retrieved_url, snapshot.observed_at,
                snapshot.source_published_at, snapshot.source_updated_at,
                snapshot.predecessor_snapshot_id,
            )
        def passages(snapshot):
            return [PassageRef(
                item.id, item.snapshot_id, snapshot.observed_at,
                item.ordinal, item.page, item.text_sha256, item.text,
            ) for item in store.get_passages(snapshot.id)]
        return compare_passages(source_ref, snapshot_ref(older), snapshot_ref(newer), passages(older), passages(newer))
