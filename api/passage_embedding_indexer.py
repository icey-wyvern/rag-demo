"""Transactional batch indexing of immutable versioned passages."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from numbers import Real

from .passage_embedding_provider import EMBEDDING_DIMENSIONS, EmbeddingProvider
from .passage_embedding_store import PassageEmbeddingStore


class EmbeddingIndexError(ValueError):
    """A batch could not be embedded or validated."""


@dataclass(frozen=True)
class EmbeddingIndexResult:
    provider: str
    model: str
    total_passages: int
    already_indexed: int
    newly_indexed: int
    remaining: int
    batches: int


def _validated_vectors(
    vectors: Sequence[Sequence[float]], expected_count: int,
) -> list[list[float]]:
    try:
        vectors = list(vectors)
    except (TypeError, ValueError) as exc:
        raise EmbeddingIndexError("Provider returned invalid vectors") from exc
    if len(vectors) != expected_count:
        raise EmbeddingIndexError(f"Provider returned {len(vectors)} vectors for {expected_count} passages")
    validated = []
    for vector in vectors:
        try:
            values = list(vector)
        except (TypeError, ValueError) as exc:
            raise EmbeddingIndexError("Provider returned a non-vector result") from exc
        if len(values) != EMBEDDING_DIMENSIONS:
            raise EmbeddingIndexError(f"Embedding must have {EMBEDDING_DIMENSIONS} dimensions")
        try:
            converted = [float(value) if isinstance(value, Real) and not isinstance(value, bool) else float("nan")
                         for value in values]
        except (OverflowError, TypeError, ValueError) as exc:
            raise EmbeddingIndexError("Embedding contains a non-finite or non-numeric value") from exc
        if any(not math.isfinite(value) for value in converted):
            raise EmbeddingIndexError("Embedding contains a non-finite or non-numeric value")
        validated.append(converted)
    return validated


class PassageEmbeddingIndexer:
    def __init__(
        self,
        session_factory,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 32,
        store_factory: Callable = PassageEmbeddingStore,
    ):
        if not provider.provider_id.strip() or not provider.model_id.strip():
            raise ValueError("Embedding provider and model identity must be non-empty")
        if provider.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(f"This index stores only {EMBEDDING_DIMENSIONS}-dimension vectors")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be an integer between 1 and 100")
        self.session_factory = session_factory
        self.provider = provider
        self.batch_size = batch_size
        self.store_factory = store_factory

    def index_missing(self) -> EmbeddingIndexResult:
        with self.session_factory.begin() as session:
            store = self.store_factory(session)
            total = store.count_passages()
            already = store.count_indexed(self.provider.provider_id, self.provider.model_id)

        indexed = batches = 0
        while True:
            with self.session_factory.begin() as session:
                store = self.store_factory(session)
                passages = store.missing_batch(
                    self.provider.provider_id, self.provider.model_id, self.batch_size,
                )
                if not passages:
                    break
                try:
                    vectors = self.provider.embed_texts([passage.text for passage in passages])
                except Exception as exc:
                    raise EmbeddingIndexError("Embedding provider failed for batch") from exc
                validated = _validated_vectors(vectors, len(passages))
                store.add_batch(
                    self.provider.provider_id,
                    self.provider.model_id,
                    [(passage.id, vector) for passage, vector in zip(passages, validated)],
                )
            indexed += len(passages)
            batches += 1

        with self.session_factory.begin() as session:
            store = self.store_factory(session)
            total = store.count_passages()
            remaining = total - store.count_indexed(
                self.provider.provider_id, self.provider.model_id,
            )
        return EmbeddingIndexResult(
            provider=self.provider.provider_id, model=self.provider.model_id,
            total_passages=total, already_indexed=already,
            newly_indexed=indexed, remaining=remaining, batches=batches,
        )
