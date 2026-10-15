"""Transactional, local-only ingestion of one observed source version."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from sqlalchemy.orm import sessionmaker

from .content_hash import sha256_bytes
from .versioned_parser import ParsedPassage, parse_pdf_passages
from .versioned_store import VersionedCorpusStore


@dataclass(frozen=True)
class ObservedSourceVersion:
    source_key: str
    canonical_url: str
    title: str
    retrieved_url: str
    content: bytes  # PDF bytes in the default parser; supplied locally by the caller.
    observed_at: datetime
    source_published_at: datetime | None = None
    source_updated_at: datetime | None = None
    effective_from: datetime | None = None

    def __post_init__(self) -> None:
        for field in ("source_key", "canonical_url", "title", "retrieved_url"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must not be blank")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("content must be non-empty bytes")
        for field in ("observed_at", "source_published_at", "source_updated_at", "effective_from"):
            value = getattr(self, field)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True)
class VersionedIngestResult:
    status: Literal["new_source", "new_snapshot", "unchanged"]
    source_id: UUID
    snapshot_id: UUID
    passage_count: int
    content_sha256: str

    @property
    def content_changed(self) -> bool:
        return self.status != "unchanged"


class VersionedIngestionService:
    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        parser: Callable[[bytes], Sequence[ParsedPassage]] = parse_pdf_passages,
        store_factory: Callable = VersionedCorpusStore,
    ):
        self.session_factory = session_factory
        self.parser = parser
        self.store_factory = store_factory

    def ingest(self, observed: ObservedSourceVersion) -> VersionedIngestResult:
        content_sha256 = sha256_bytes(observed.content)
        # sessionmaker.begin() commits once on success and rolls back on any exception.
        with self.session_factory.begin() as session:
            store = self.store_factory(session)
            source = store.get_source(observed.source_key)
            source_created = source is None
            if source is None:
                source = store.add_source(observed.source_key, observed.canonical_url, observed.title)
            else:
                existing = store.get_snapshot_by_hash(source.id, content_sha256)
                if existing is not None:
                    return VersionedIngestResult(
                        status="unchanged",
                        source_id=source.id,
                        snapshot_id=existing.id,
                        passage_count=store.passage_count(existing.id),
                        content_sha256=content_sha256,
                    )

            predecessor = store.get_latest_snapshot(source.id)
            passages = list(self.parser(observed.content))
            if not passages or any(not passage.text for passage in passages):
                raise ValueError("Observed content produced no usable passages")

            if not source_created:
                store.update_source_metadata(source, observed.canonical_url, observed.title)
            snapshot = store.add_snapshot(
                source_id=source.id,
                predecessor_snapshot_id=predecessor.id if predecessor else None,
                retrieved_url=observed.retrieved_url,
                content_sha256=content_sha256,
                observed_at=observed.observed_at,
                source_published_at=observed.source_published_at,
                source_updated_at=observed.source_updated_at,
                effective_from=observed.effective_from,
            )
            store.add_passages(snapshot.id, passages)
            return VersionedIngestResult(
                status="new_source" if source_created else "new_snapshot",
                source_id=source.id,
                snapshot_id=snapshot.id,
                passage_count=len(passages),
                content_sha256=content_sha256,
            )
