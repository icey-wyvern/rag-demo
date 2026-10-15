"""Validated, API-independent retrieval service for the versioned corpus."""

from collections.abc import Callable
from datetime import datetime
from typing import Literal

from .evidence import Evidence
from .hybrid_rrf import DEFAULT_RRF_CONSTANT, candidate_depth, fuse_rrf
from .passage_embedding_provider import EMBEDDING_DIMENSIONS, EmbeddingProvider, validate_embedding_vectors
from .versioned_retrieval_store import VersionedRetrievalStore


DEFAULT_K = 5
MAX_K = 20
RetrievalMode = Literal["lexical", "semantic", "hybrid"]


class SemanticRetrievalError(ValueError):
    """The selected semantic provider/model cannot answer this query."""


class VersionedRetriever:
    def __init__(
        self, session_factory, *,
        embedding_provider: EmbeddingProvider | None = None,
        store_factory: Callable = VersionedRetrievalStore,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
    ):
        if embedding_provider is not None:
            if embedding_provider.dimensions != EMBEDDING_DIMENSIONS:
                raise ValueError(f"Semantic retrieval requires {EMBEDDING_DIMENSIONS}-dimension vectors")
            if not embedding_provider.provider_id.strip() or not embedding_provider.model_id.strip():
                raise ValueError("Embedding provider and model identity must be non-empty")
        if isinstance(rrf_constant, bool) or not isinstance(rrf_constant, int) or not 1 <= rrf_constant <= 100:
            raise ValueError("RRF constant must be an integer between 1 and 100")
        self.session_factory = session_factory
        self.embedding_provider = embedding_provider
        self.store_factory = store_factory
        self.rrf_constant = rrf_constant

    def search(
        self, question: str, *, mode: RetrievalMode = "lexical",
        as_of: datetime | None = None, k: int = DEFAULT_K,
    ) -> list[Evidence]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty")
        if mode not in ("lexical", "semantic", "hybrid"):
            raise ValueError("mode must be lexical, semantic, or hybrid")
        if as_of is not None and (not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None):
            raise ValueError("as_of must be timezone-aware")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be an integer between 1 and {MAX_K}")
        with self.session_factory() as session:
            store = self.store_factory(session)
            query = question.strip()
            if mode == "lexical":
                return store.search(query, as_of, k)
            if mode == "hybrid":
                store.begin_consistent_read()
            provider = self.embedding_provider
            if provider is None:
                raise SemanticRetrievalError("Semantic retrieval requires an embedding provider")
            if not store.has_embeddings(provider.provider_id, provider.model_id):
                raise SemanticRetrievalError(
                    f"No indexed passages for provider={provider.provider_id} model={provider.model_id}"
                )
            try:
                raw_vectors = provider.embed_texts([query])
            except Exception as exc:
                raise SemanticRetrievalError("Query embedding provider failed") from exc
            try:
                vector = validate_embedding_vectors(raw_vectors, 1)[0]
            except ValueError as exc:
                raise SemanticRetrievalError(f"Invalid query embedding: {exc}") from exc
            if mode == "semantic":
                return store.semantic_search(vector, provider.provider_id, provider.model_id, as_of, k)
            depth = candidate_depth(k)
            lexical = store.search(query, as_of, depth)
            semantic = store.semantic_search(vector, provider.provider_id, provider.model_id, as_of, depth)
            return fuse_rrf(lexical, semantic, k=k, constant=self.rrf_constant)
