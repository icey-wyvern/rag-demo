"""SQLAlchemy operations for the versioned corpus; transaction owned by the caller."""

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .versioned_models import Passage, Source, SourceSnapshot
from .versioned_parser import ParsedPassage
from .content_hash import sha256_text


class VersionedCorpusStore:
    def __init__(self, session: Session):
        self.session = session

    def get_source(self, source_key: str) -> Source | None:
        return self.session.scalar(select(Source).where(Source.source_key == source_key))

    def add_source(self, source_key: str, canonical_url: str, title: str) -> Source:
        source = Source(source_key=source_key, canonical_url=canonical_url, title=title)
        self.session.add(source)
        self.session.flush()
        return source

    def update_source_metadata(self, source: Source, canonical_url: str, title: str) -> None:
        if source.canonical_url != canonical_url or source.title != title:
            source.canonical_url = canonical_url
            source.title = title
            self.session.flush()

    def get_snapshot_by_hash(self, source_id: UUID, content_sha256: str) -> SourceSnapshot | None:
        return self.session.scalar(
            select(SourceSnapshot).where(
                SourceSnapshot.source_id == source_id,
                SourceSnapshot.content_sha256 == content_sha256,
            )
        )

    def get_latest_snapshot(self, source_id: UUID) -> SourceSnapshot | None:
        return self.session.scalar(
            select(SourceSnapshot)
            .where(SourceSnapshot.source_id == source_id)
            .order_by(
                SourceSnapshot.observed_at.desc(),
                SourceSnapshot.created_at.desc(),
                SourceSnapshot.id.desc(),
            )
            .limit(1)
        )

    def add_snapshot(
        self,
        *,
        source_id: UUID,
        predecessor_snapshot_id: UUID | None,
        retrieved_url: str,
        content_sha256: str,
        observed_at: datetime,
        source_published_at: datetime | None,
        source_updated_at: datetime | None,
        effective_from: datetime | None,
    ) -> SourceSnapshot:
        snapshot = SourceSnapshot(
            source_id=source_id,
            predecessor_snapshot_id=predecessor_snapshot_id,
            retrieved_url=retrieved_url,
            content_sha256=content_sha256,
            observed_at=observed_at,
            source_published_at=source_published_at,
            source_updated_at=source_updated_at,
            effective_from=effective_from,
        )
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def add_passages(self, snapshot_id: UUID, passages: Sequence[ParsedPassage]) -> None:
        self.session.add_all(
            [
                Passage(
                    snapshot_id=snapshot_id,
                    ordinal=ordinal,
                    page=passage.page,
                    text=passage.text,
                    text_sha256=sha256_text(passage.text),
                )
                for ordinal, passage in enumerate(passages)
            ]
        )
        self.session.flush()

    def passage_count(self, snapshot_id: UUID) -> int:
        return self.session.scalar(
            select(func.count(Passage.id)).where(Passage.snapshot_id == snapshot_id)
        )
