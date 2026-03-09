from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import httpx
from sqlalchemy.orm import Session

from .chunker import make_chunks
from .config import get_settings, validate_ingest_settings
from .embeddings import get_embeddings
from .pdf_loader import load_pdf, load_pdf_from_bytes
from .seed_docs import SEED_PDFS
from .store import (
    corpus_is_ready,
    count_ready_documents,
    get_or_create_corpus_state,
    mark_corpus_state,
    mark_all_documents_pending,
    release_corpus_lock,
    replace_document_chunks,
    try_acquire_corpus_lock,
    upsert_document,
)


@dataclass(frozen=True)
class IngestResult:
    status: str
    documents_total: int
    documents_ready: int
    message: str = ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_path(cache_dir: Path, base_path: str) -> Path:
    return cache_dir / f"{base_path}.pdf"


def _load_cached_or_local_pdf(*, base_path: str, docs_dir: Path, cache_dir: Path):
    local_path = docs_dir / f"{base_path}.pdf"
    if local_path.exists():
        return load_pdf(local_path), local_path.read_bytes()

    cached_path = _cache_path(cache_dir, base_path)
    if cached_path.exists():
        return load_pdf(cached_path), cached_path.read_bytes()

    return None, None


async def _download_seed_pdf(client: httpx.AsyncClient, *, url: str, base_path: str, cache_dir: Path):
    last_error: Exception | None = None
    for _ in range(3):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
            cache_dir.mkdir(parents=True, exist_ok=True)
            _cache_path(cache_dir, base_path).write_bytes(data)
            return load_pdf_from_bytes(name=url.rsplit("/", 1)[-1] or base_path, data=data), data
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(1.0)
    raise RuntimeError(f"Failed to download seed document {base_path}: {last_error}")


def _build_pages_and_texts(paragraph_items: List[Dict[str, object]]) -> List[tuple[int, str]]:
    page_buckets: Dict[int, List[str]] = {}
    for item in paragraph_items:
        page = int(item.get("page", 1))
        text = str(item.get("text", ""))
        if text:
            page_buckets.setdefault(page, []).append(text)

    pages_and_texts: List[tuple[int, str]] = []
    for page, paragraphs in sorted(page_buckets.items()):
        for _, chunk_text in make_chunks(paragraphs):
            pages_and_texts.append((page, chunk_text))
    return pages_and_texts


async def ingest_seed_docs(db: Session, *, force: bool = False) -> IngestResult:
    settings = get_settings()
    validate_ingest_settings(settings)
    documents_total = len(SEED_PDFS)
    if not force and corpus_is_ready(db, ingest_version=settings.CORPUS_VERSION, documents_total=documents_total):
        return IngestResult(
            status="ready",
            documents_total=documents_total,
            documents_ready=documents_total,
            message="Corpus already ready.",
        )

    if not try_acquire_corpus_lock(db):
        state = get_or_create_corpus_state(
            db,
            ingest_version=settings.CORPUS_VERSION,
            documents_total=documents_total,
        )
        db.commit()
        return IngestResult(
            status=state.status,
            documents_total=state.documents_total,
            documents_ready=state.documents_ready,
            message="Another ingestion process is already running.",
        )

    docs_dir = Path(settings.SEED_DOCS_DIR)
    cache_dir = Path(settings.SEED_CACHE_DIR)
    mark_corpus_state(
        db,
        status="running",
        ingest_version=settings.CORPUS_VERSION,
        documents_total=documents_total,
        documents_ready=0,
        last_error=None,
    )
    mark_all_documents_pending(db, settings.CORPUS_VERSION)
    db.commit()

    try:
        timeout = httpx.Timeout(settings.SEED_DOWNLOAD_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            for seed in SEED_PDFS:
                title = seed["title"]
                base_path = seed["base_path"]
                url = seed["url"]

                doc_payload, data = _load_cached_or_local_pdf(
                    base_path=base_path,
                    docs_dir=docs_dir,
                    cache_dir=cache_dir,
                )
                if doc_payload is None or data is None:
                    doc_payload, data = await _download_seed_pdf(
                        client,
                        url=url,
                        base_path=base_path,
                        cache_dir=cache_dir,
                    )

                pages_and_texts = _build_pages_and_texts(doc_payload.get("paragraphs", []))
                if not pages_and_texts:
                    raise RuntimeError(f"No extractable text found in {title}.")

                embeddings = get_embeddings().embed_documents([text for _, text in pages_and_texts])
                source_sha256 = _sha256_bytes(data)

                doc = upsert_document(
                    db,
                    title=title,
                    base_path=base_path,
                    url=url,
                    source_sha256=source_sha256,
                    ingest_version=settings.CORPUS_VERSION,
                    status="indexing",
                    chunk_count=len(pages_and_texts),
                )
                replace_document_chunks(
                    db,
                    doc_id=doc.id,
                    pages_and_texts=pages_and_texts,
                    embeddings=embeddings,
                )
                upsert_document(
                    db,
                    title=title,
                    base_path=base_path,
                    url=url,
                    source_sha256=source_sha256,
                    ingest_version=settings.CORPUS_VERSION,
                    status="ready",
                    chunk_count=len(pages_and_texts),
                )
                ready_count = count_ready_documents(db, settings.CORPUS_VERSION)
                mark_corpus_state(
                    db,
                    status="running",
                    ingest_version=settings.CORPUS_VERSION,
                    documents_total=documents_total,
                    documents_ready=ready_count,
                    last_error=None,
                )
                db.commit()

        mark_corpus_state(
            db,
            status="ready",
            ingest_version=settings.CORPUS_VERSION,
            documents_total=documents_total,
            documents_ready=documents_total,
            last_error=None,
        )
        db.commit()
        return IngestResult(
            status="ready",
            documents_total=documents_total,
            documents_ready=documents_total,
            message="Corpus ingestion completed.",
        )
    except Exception as exc:
        ready_count = count_ready_documents(db, settings.CORPUS_VERSION)
        mark_corpus_state(
            db,
            status="error",
            ingest_version=settings.CORPUS_VERSION,
            documents_total=documents_total,
            documents_ready=ready_count,
            last_error=str(exc),
        )
        db.commit()
        return IngestResult(
            status="error",
            documents_total=documents_total,
            documents_ready=ready_count,
            message=str(exc),
        )
    finally:
        release_corpus_lock(db)
        db.commit()


def run_ingest(*, force: bool = False) -> IngestResult:
    from .db import SessionLocal

    db = SessionLocal()
    try:
        return asyncio.run(ingest_seed_docs(db, force=force))
    finally:
        db.close()
