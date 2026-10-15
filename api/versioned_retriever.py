"""Validated, API-independent retrieval service for the versioned corpus."""

from collections.abc import Callable
from datetime import datetime

from .evidence import Evidence
from .versioned_retrieval_store import VersionedRetrievalStore


DEFAULT_K = 5
MAX_K = 20


class VersionedRetriever:
    def __init__(self, session_factory, *, store_factory: Callable = VersionedRetrievalStore):
        self.session_factory = session_factory
        self.store_factory = store_factory

    def search(self, question: str, *, as_of: datetime | None = None, k: int = DEFAULT_K) -> list[Evidence]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty")
        if as_of is not None and (not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None):
            raise ValueError("as_of must be timezone-aware")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be an integer between 1 and {MAX_K}")
        with self.session_factory() as session:
            return self.store_factory(session).search(question.strip(), as_of, k)
