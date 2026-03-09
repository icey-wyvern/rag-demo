from typing import Any, Dict, List

from sqlalchemy.orm import Session

from .embeddings import get_embeddings
from .store import vector_search


def similarity_search(db: Session, query: str, k: int = 8) -> List[Dict[str, Any]]:
    query_vec = get_embeddings().embed_query(query)
    return vector_search(db, query_vec=query_vec, k=k)
