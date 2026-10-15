"""Small provider boundary for versioned passage embeddings."""

from collections.abc import Sequence
from typing import Protocol


EMBEDDING_DIMENSIONS = 1536


class EmbeddingProvider(Protocol):
    provider_id: str
    model_id: str
    dimensions: int

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


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
