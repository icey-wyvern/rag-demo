"""Immutable retrieval result for the versioned corpus."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class Evidence:
    source_id: UUID
    source_key: str
    source_title: str
    canonical_url: str
    snapshot_id: UUID
    snapshot_content_sha256: str
    retrieved_url: str
    observed_at: datetime
    source_published_at: datetime | None
    source_updated_at: datetime | None
    passage_id: UUID
    passage_ordinal: int
    page: int | None
    text: str
    text_sha256: str
    retrieval_method: str
    score: float
    rank: int
    score_kind: str = "lexical_ts_rank_cd"
    lexical_rank: int | None = None
    semantic_rank: int | None = None
