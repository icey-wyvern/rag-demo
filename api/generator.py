from typing import List, Tuple, Dict
from langchain_openai import ChatOpenAI
from .config import get_settings

settings = get_settings()

class Generator:
	"""
	Grounded answer synthesis over retrieved chunks.
	Returns (answer_text, citations[{title,url}...]).
	"""
	def __init__(self, model: str | None = None, temperature: float = 0.2):
		self._llm = ChatOpenAI(
			model=model or settings.GEN_MODEL,
			openai_api_key=settings.OPENAI_API_KEY,
			temperature=temperature,
		)

	def _build_context(self, contexts: List[Dict]) -> str:
		# Enumerate sources so the model can reference them if needed.
		lines: List[str] = []
		for i, c in enumerate(contexts, start=1):
			title = c.get("title", "")
			url = c.get("url", "")
			content = c.get("content", "")
			lines.append(f"[{i}] {title}\n{url}\n{content}\n")
		return "\n".join(lines)

	def answer(self, question: str, contexts: List[Dict]) -> Tuple[str, List[Dict[str, str]]]:
		"""
		question: user query
		contexts: list of {title, url, content, ...} (top-k retrieved)
		"""
		if not contexts:
			return ("I can’t answer that from the current context.", [])

		context_txt = self._build_context(contexts)

		system = (
			"You are a careful assistant. Answer ONLY using the provided context. "
			"If the context is insufficient, say so briefly. Keep answers concise and clear."
		)
		user = (
			f"Question:\n{question}\n\n"
			"Context passages (numbered):\n"
			f"{context_txt}\n\n"
			"Write a short answer grounded in the context above."
		)

		resp = self._llm.invoke([{"role": "system", "content": system},
		                         {"role": "user", "content": user}])
		answer_text = (resp.content or "").strip()

		# Simple citation list: unique (title,url) from provided contexts (preserve order)
		seen = set()
		citations: List[Dict[str, str]] = []
		for c in contexts:
			key = (c.get("title", ""), c.get("url", ""))
			if key not in seen:
				seen.add(key)
				citations.append({"title": key[0], "url": key[1]})
		return answer_text, citations[:5]
