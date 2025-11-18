from typing import List
from langchain_openai import OpenAIEmbeddings
from .config import get_settings

settings = get_settings()

class Embedder:
	"""Thin wrapper around OpenAI embeddings for batch embedding."""
	def __init__(self, model: str | None = None):
		self._model = model or settings.EMBED_MODEL
		self._emb = OpenAIEmbeddings(model=self._model, openai_api_key=settings.OPENAI_API_KEY)

	def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
		out: List[List[float]] = []
		buf: List[str] = []
		for t in texts or []:
			buf.append(t or "")
			if len(buf) >= batch_size:
				out.extend(self._emb.embed_documents(buf))
				buf.clear()
		if buf:
			out.extend(self._emb.embed_documents(buf))
		return out
