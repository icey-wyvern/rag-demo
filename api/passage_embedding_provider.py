"""Small provider boundary for versioned passage embeddings."""

from collections.abc import Sequence
import math
from numbers import Real
from typing import Protocol


EMBEDDING_DIMENSIONS = 1536


class EmbeddingProvider(Protocol):
    provider_id: str
    model_id: str
    dimensions: int

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def validate_embedding_vectors(
    vectors: Sequence[Sequence[float]], expected_count: int,
) -> list[list[float]]:
    """Validate ordered provider output for the repository's vector(1536) column."""
    try:
        vectors = list(vectors)
    except (TypeError, ValueError) as exc:
        raise ValueError("Provider returned invalid vectors") from exc
    if len(vectors) != expected_count:
        raise ValueError(f"Provider returned {len(vectors)} vectors for {expected_count} passages")
    validated = []
    for vector in vectors:
        try:
            values = list(vector)
        except (TypeError, ValueError) as exc:
            raise ValueError("Provider returned a non-vector result") from exc
        if len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Embedding must have {EMBEDDING_DIMENSIONS} dimensions")
        try:
            converted = [float(value) if isinstance(value, Real) and not isinstance(value, bool) else float("nan")
                         for value in values]
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("Embedding contains a non-finite or non-numeric value") from exc
        if any(not math.isfinite(value) for value in converted):
            raise ValueError("Embedding contains a non-finite or non-numeric value")
        if not any(value != 0.0 for value in converted):
            raise ValueError("Cosine embedding must not be a zero vector")
        validated.append(converted)
    return validated


class OpenAIPassageEmbeddingProvider:
    """Opt-in OpenAI adapter; no client is made until indexing needs a batch."""

    provider_id = "openai"
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, *, model: str, api_key: str, timeout_seconds: float = 30.0):
        if not model.strip() or not api_key.strip() or timeout_seconds <= 0:
            raise ValueError("OpenAI embedding model, API key, and positive timeout are required")
        self.model_id = model
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client = None

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        from langchain_openai import OpenAIEmbeddings

        if self._client is None:
            self._client = OpenAIEmbeddings(
                model=self.model_id,
                dimensions=self.dimensions,
                openai_api_key=self._api_key,
                timeout=self.timeout_seconds,
                max_retries=2,
            )
        return self._client.embed_documents(list(texts))
