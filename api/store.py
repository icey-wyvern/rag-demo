from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, bindparam, func, select, text
from sqlalchemy.orm import Session

from .models import Chunk, CorpusState, Document


CORPUS_LOCK_ID = 5_250_001


def get_or_create_corpus_state(db: Session, ingest_version: int, documents_total: int) -> CorpusState:
    state = db.get(CorpusState, 1)
    if state is None:
        state = CorpusState(
            id=1,
            status="pending",
            ingest_version=ingest_version,
            documents_total=documents_total,
            documents_ready=0,
            last_error=None,
        )
        db.add(state)
        db.flush()
        return state

    state.ingest_version = ingest_version
    state.documents_total = documents_total
    db.add(state)
    db.flush()
    return state


def mark_corpus_state(
    db: Session,
    *,
    status: str,
    ingest_version: int,
    documents_total: int,
    documents_ready: int,
    last_error: str | None = None,
) -> CorpusState:
    state = get_or_create_corpus_state(db, ingest_version=ingest_version, documents_total=documents_total)
    state.status = status
    state.documents_ready = documents_ready
    state.last_error = last_error
    db.add(state)
    db.flush()
    return state


def get_document_by_base_path(db: Session, base_path: str) -> Optional[Document]:
    return db.execute(select(Document).where(Document.base_path == base_path)).scalars().first()


def upsert_document(
    db: Session,
    *,
    title: str,
    base_path: str,
    url: str,
    source_sha256: str | None,
    ingest_version: int,
    status: str,
    chunk_count: int,
) -> Document:
    doc = get_document_by_base_path(db, base_path)
    if doc is None:
        doc = Document(
            title=title,
            base_path=base_path,
            url=url,
            source_sha256=source_sha256,
            ingest_version=ingest_version,
            status=status,
            chunk_count=chunk_count,
        )
    else:
        doc.title = title
        doc.url = url
        doc.source_sha256 = source_sha256
        doc.ingest_version = ingest_version
        doc.status = status
        doc.chunk_count = chunk_count

    db.add(doc)
    db.flush()
    return doc


def count_ready_documents(db: Session, ingest_version: int) -> int:
    return db.execute(
        select(func.count())
        .select_from(Document)
        .where(Document.status == "ready", Document.ingest_version == ingest_version)
    ).scalar_one()


def mark_all_documents_pending(db: Session, ingest_version: int) -> None:
    for document in db.query(Document).all():
        document.status = "pending"
        document.ingest_version = ingest_version
        db.add(document)
    db.flush()


def replace_document_chunks(
    db: Session,
    *,
    doc_id,
    pages_and_texts: Sequence[tuple[int, str]],
    embeddings: Sequence[Sequence[float]],
) -> None:
    if len(pages_and_texts) != len(embeddings):
        raise ValueError("pages_and_texts and embeddings length mismatch")

    db.query(Chunk).filter(Chunk.doc_id == doc_id).delete(synchronize_session=False)

    db.add_all(
        [
            Chunk(doc_id=doc_id, ord=index, page=page, content=content, embedding=embedding)
            for index, ((page, content), embedding) in enumerate(zip(pages_and_texts, embeddings))
        ]
    )
    db.flush()


def vector_search(db: Session, query_vec: List[float], k: int = 8) -> List[dict]:
    rows = db.execute(
        text(
            """
            SELECT
                c.id AS chunk_id,
                c.page AS page,
                c.content AS content,
                d.id AS doc_id,
                d.title AS title,
                d.url AS url,
                (c.embedding <-> :qvec) AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE d.status = 'ready'
            ORDER BY c.embedding <-> :qvec
            LIMIT :k
            """
        ).bindparams(
            bindparam("qvec", value=query_vec, type_=Vector(1536)),
            bindparam("k", value=int(k), type_=Integer),
        )
    ).mappings()
    return [dict(row) for row in rows]


def try_acquire_corpus_lock(db: Session) -> bool:
    return bool(db.execute(text("SELECT pg_try_advisory_lock(:id)"), {"id": CORPUS_LOCK_ID}).scalar())


def release_corpus_lock(db: Session) -> None:
    db.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": CORPUS_LOCK_ID})


def corpus_is_ready(db: Session, *, ingest_version: int, documents_total: int) -> bool:
    state = db.get(CorpusState, 1)
    if state is None:
        return False
    return (
        state.status == "ready"
        and state.ingest_version == ingest_version
        and state.documents_total == documents_total
        and state.documents_ready == documents_total
    )


def iter_ready_documents(db: Session) -> Iterable[Document]:
    return db.execute(select(Document).where(Document.status == "ready")).scalars()


def reset_corpus(db: Session) -> None:
    db.query(Chunk).delete(synchronize_session=False)
    db.query(Document).delete(synchronize_session=False)
    db.query(CorpusState).delete(synchronize_session=False)
    db.flush()
