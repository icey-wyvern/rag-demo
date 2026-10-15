"""SQLAlchemy persistence for model-specific passage vectors."""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from .versioned_models import Passage, PassageEmbedding


@dataclass(frozen=True)
class PassageToEmbed:
    id: UUID
    text: str


class PassageEmbeddingStore:
    def __init__(self, session: Session):
        self.session = session

    def count_passages(self) -> int:
        return self.session.scalar(select(func.count(Passage.id)))

    def count_indexed(self, provider: str, model: str) -> int:
        return self.session.scalar(
            select(func.count(PassageEmbedding.id)).where(
                PassageEmbedding.provider == provider, PassageEmbedding.model == model,
            )
        )

    def missing_batch(self, provider: str, model: str, limit: int) -> list[PassageToEmbed]:
        # Lock passage rows while embedding the bounded batch. Concurrent
        # indexers skip claimed rows; the unique constraint remains the final guard.
        rows = self.session.execute(
            select(Passage.id, Passage.text)
            .outerjoin(
                PassageEmbedding,
                and_(
                    PassageEmbedding.passage_id == Passage.id,
                    PassageEmbedding.provider == provider,
                    PassageEmbedding.model == model,
                ),
            )
            .where(PassageEmbedding.id.is_(None))
            .order_by(Passage.id)
            .limit(limit)
            .with_for_update(of=Passage, skip_locked=True)
        ).all()
        return [PassageToEmbed(id=row.id, text=row.text) for row in rows]

    def add_batch(
        self,
        provider: str,
        model: str,
        vectors: Sequence[tuple[UUID, list[float]]],
    ) -> None:
        self.session.add_all([
            PassageEmbedding(
                passage_id=passage_id, provider=provider, model=model,
                dimensions=len(vector), embedding=vector,
            )
            for passage_id, vector in vectors
        ])
        self.session.flush()
