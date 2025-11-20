from typing import Literal, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import SessionLocal
from .ingest import needs_ingest, ingest_seed_docs
from .config import get_settings
from .vectorstore import get_vectorstore
from .generator import Generator

settings = get_settings()
app = FastAPI(title="gvt-rag")


class QueryRequest(BaseModel):
	q: str
	mode: Literal["chunks", "generate"] = "chunks"


def _snippet(txt: str, limit: int = 360) -> str:
	if not txt:
		return ""
	return txt if len(txt) <= limit else txt[: limit - 1] + "…"


@app.on_event("startup")
async def _startup_ingest() -> None:
	db = SessionLocal()
	try:
		if needs_ingest(db):
			await ingest_seed_docs(db)
	finally:
		db.close()


@app.get("/health")
def health():
	return {"ok": True}


@app.post("/api/query")
async def query(req: QueryRequest):
	q = (req.q or "").strip()
	if not q:
		raise HTTPException(status_code=400, detail="Empty query")

	vstore = get_vectorstore()
	# Plain similarity search, k=8
	docs = vstore.similarity_search(q, k=8)

	if req.mode == "chunks":
		chunks = []
		for d in docs:
			meta = d.metadata or {}
			page = meta.get("page")
			try:
				page = int(page) if page is not None else None
			except (TypeError, ValueError):
				page = None

			chunks.append(
				{
					"title": meta.get("title", ""),
					"url": meta.get("url", ""),
					"page": page,
					"snippet": _snippet(d.page_content),
				}
			)
		return {"chunks": chunks}

	# Generative (non-streaming)
	if not settings.OPENAI_API_KEY:
		raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")

	contexts: List[Dict] = []
	seen = set()
	for d in docs:
		meta = d.metadata or {}
		title = meta.get("title", "")
		url = meta.get("url", "")
		page = meta.get("page")
		try:
			page = int(page) if page is not None else None
		except (TypeError, ValueError):
			page = None

		key = (title, url, page, d.page_content)
		if key in seen:
			continue
		seen.add(key)

		contexts.append(
			{
				"title": title,
				"url": url,
				"page": page,
				"content": d.page_content,
			}
		)

	gen = Generator()
	answer, citations = gen.answer(q, contexts)
	return {"answer": answer, "citations": citations}


# Mount static AFTER routes so /api/* isn’t shadowed by StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")
