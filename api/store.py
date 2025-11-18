from typing import List, Tuple, Optional
from sqlalchemy import select, text, bindparam, Integer
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

from .models import Document, Chunk

# ---------- Document ops ----------

def get_document_by_base_path(db: Session, base_path: str) -> Optional[Document]:
	return db.execute(
		select(Document).where(Document.base_path == base_path)
	).scalars().first()

def upsert_document(db: Session, title: str, base_path: str, url: str) -> Document:
	doc = get_document_by_base_path(db, base_path)
	if doc:
		changed = False
		if doc.title != title:
			doc.title = title
			changed = True
		if doc.url != url:
			doc.url = url
			changed = True
		if changed:
			db.add(doc)
			db.flush()
		return doc
	doc = Document(title=title, base_path=base_path, url=url)
	db.add(doc)
	db.flush()
	return doc

def count_documents(db: Session) -> int:
	return db.execute(text("SELECT COUNT(*) FROM documents")).scalar_one()

# ---------- Chunk ops ----------

def insert_chunks(db: Session, doc_id, ordered_texts: List[Tuple[int, str]], embeddings: List[List[float]]) -> None:
	if not ordered_texts:
		return
	if len(ordered_texts) != len(embeddings):
		raise ValueError("ordered_texts and embeddings length mismatch")
	objects = []
	for (ord_idx, content), emb in zip(ordered_texts, embeddings):
		objects.append(Chunk(doc_id=doc_id, ord=ord_idx, content=content, embedding=emb))
	db.add_all(objects)
	db.flush()

def delete_chunks_for_document(db: Session, doc_id) -> None:
	db.query(Chunk).filter(Chunk.doc_id == doc_id).delete(synchronize_session=False)
	db.flush()

# ---------- Vector search ----------

def vector_search(db: Session, query_vec: List[float], k: int = 8) -> List[dict]:
	"""
	Return top-k chunks with their document metadata.
	Bind the query vector as pgvector so <-> works.
	"""
	sql = text("""
		SELECT
			c.id        AS chunk_id,
			c.ord       AS ord,
			c.content   AS content,
			d.id        AS doc_id,
			d.title     AS title,
			d.url       AS url,
			(c.embedding <-> :qvec) AS distance
		FROM chunks c
		JOIN documents d ON d.id = c.doc_id
		ORDER BY c.embedding <-> :qvec
		LIMIT :k
	""").bindparams(
		bindparam("qvec", value=query_vec, type_=Vector(1536)),
		bindparam("k", value=int(k), type_=Integer),
	)

	rows = db.execute(sql).mappings().all()
	return [dict(r) for r in rows]
