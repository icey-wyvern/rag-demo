from typing import List, Tuple, Dict, Any
from langchain_openai import ChatOpenAI
from .config import get_settings

settings = get_settings()


class Generator:
	"""
	Grounded answer synthesis over retrieved chunks.
	Returns (answer_text, citations[{title, url, page?}...]).
	"""

	def __init__(self, model: str | None = None, temperature: float = 0.2):
		self._llm = ChatOpenAI(
			model=model or settings.GEN_MODEL,
			openai_api_key=settings.OPENAI_API_KEY,
			temperature=temperature,
		)

	def _build_context(self, contexts: List[Dict[str, Any]]) -> str:
		"""
		Build a single big context string for the LLM.

		Each context dict is expected to look like:
			{
				"title": str,
				"url": str,
				"page": int | None,
				"content": str,
				...
			}
		"""
		lines: List[str] = []
		for i, c in enumerate(contexts, start=1):
			title = c.get("title", "")
			url = c.get("url", "")
			content = c.get("content", "")

			page = c.get("page")
			try:
				page_num = int(page) if page is not None else None
			except (TypeError, ValueError):
				page_num = None

			if page_num is not None:
				header = f"[{i}] {title} (page {page_num})"
			else:
				header = f"[{i}] {title}"

			lines.append(f"{header}\n{url}\n{content}\n")

		return "\n".join(lines)

	def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
		"""
		question: user query
		contexts: list of {title, url, page?, content, ...} (top-k retrieved)

		Returns:
			(answer_text, citations)

		where citations is a list of:
			{
				"title": str,
				"url": str,
				"page": int (optional)
			}
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

		resp = self._llm.invoke(
			[
				{"role": "system", "content": system},
				{"role": "user", "content": user},
			]
		)
		answer_text = (resp.content or "").strip()

		# Simple citation list: unique (title, url, page) from provided contexts (preserve order)
		seen = set()
		citations: List[Dict[str, Any]] = []

		for c in contexts:
			title = c.get("title", "")
			url = c.get("url", "")

			page = c.get("page")
			try:
				page_num = int(page) if page is not None else None
			except (TypeError, ValueError):
				page_num = None

			key = (title, url, page_num)
			if key in seen:
				continue
			seen.add(key)

			citation: Dict[str, Any] = {
				"title": title,
				"url": url,
			}
			if page_num is not None:
				citation["page"] = page_num

			citations.append(citation)

		# Return at most 5 citations
		return answer_text, citations[:5]
