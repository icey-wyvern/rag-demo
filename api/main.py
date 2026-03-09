from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import get_settings
from .db import SessionLocal
from .generator import Generator
from .models import CorpusState
from .vectorstore import similarity_search

settings = get_settings()
app = FastAPI(title="gvt-rag")
static_dir = Path(__file__).resolve().parents[1] / "static"


class QueryRequest(BaseModel):
    q: str
    mode: Literal["chunks", "generate"] = "chunks"


def _snippet(text: str, limit: int = 360) -> str:
    if not text or len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "page": doc.get("page"),
        "snippet": _snippet(doc.get("content", "")),
        "content": doc.get("content", ""),
    }


def _get_corpus_state(db: Session) -> Dict[str, Any]:
    state = db.get(CorpusState, 1)
    if state is None:
        return {
            "status": "pending",
            "ingest_version": settings.CORPUS_VERSION,
            "documents_total": 0,
            "documents_ready": 0,
            "last_error": None,
            "is_ready": False,
        }
    return {
        "status": state.status,
        "ingest_version": state.ingest_version,
        "documents_total": state.documents_total,
        "documents_ready": state.documents_ready,
        "last_error": state.last_error,
        "is_ready": state.status == "ready" and state.documents_ready == state.documents_total,
    }


@app.get("/health")
def health():
    db = SessionLocal()
    try:
        corpus = _get_corpus_state(db)
        return {"ok": True, "corpus": corpus}
    finally:
        db.close()


@app.get("/api/status")
def corpus_status():
    db = SessionLocal()
    try:
        return _get_corpus_state(db)
    finally:
        db.close()


@app.post("/api/query")
async def query(req: QueryRequest):
    q = (req.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    db = SessionLocal()
    try:
        corpus = _get_corpus_state(db)
        if not corpus["is_ready"]:
            detail = "Corpus is not ready yet."
            if corpus["status"] == "error" and corpus["last_error"]:
                detail = f"Corpus ingestion failed: {corpus['last_error']}"
            elif corpus["status"] == "running":
                detail = "Corpus ingestion is still running."
            raise HTTPException(status_code=503, detail=detail)

        try:
            docs = similarity_search(db, q, k=8)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Document store unavailable: {exc}") from exc

        serialized_docs = [_serialize_doc(doc) for doc in docs]
        chunks = [{key: value for key, value in doc.items() if key != "content"} for doc in serialized_docs]

        if req.mode == "chunks":
            return {"chunks": chunks}

        contexts: List[Dict[str, Any]] = []
        seen = set()
        for doc in serialized_docs:
            key = (doc["title"], doc["url"], doc["page"], doc["content"])
            if key in seen:
                continue
            seen.add(key)
            contexts.append(
                {
                    "title": doc["title"],
                    "url": doc["url"],
                    "page": doc["page"],
                    "content": doc["content"],
                }
            )

        answer, citations = Generator().answer(q, contexts)
        return {"answer": answer, "citations": citations, "chunks": chunks}
    finally:
        db.close()

if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
else:
    @app.get("/")
    def root():
        return {
            "name": "gvt-rag",
            "ui_ready": False,
            "message": "Frontend assets have not been built yet.",
        }
