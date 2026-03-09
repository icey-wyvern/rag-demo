from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI

from .config import get_settings, validate_generation_settings

settings = get_settings()


class Generator:
    def __init__(self, model: str | None = None, temperature: float = 0.2):
        validate_generation_settings(settings)
        self._llm = ChatOpenAI(
            model=model or settings.GEN_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )

    def _build_context(self, contexts: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for index, context in enumerate(contexts, start=1):
            title = context.get("title", "")
            url = context.get("url", "")
            content = context.get("content", "")

            page = context.get("page")
            try:
                page_num = int(page) if page is not None else None
            except (TypeError, ValueError):
                page_num = None

            header = f"[{index}] {title}"
            if page_num is not None:
                header = f"{header} (page {page_num})"
            lines.append(f"{header}\n{url}\n{content}\n")

        return "\n".join(lines)

    def answer(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not contexts:
            return ("I can't answer that from the current context.", [])

        resp = self._llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a careful assistant. Answer only using the provided context. "
                        "If the context is insufficient, say so briefly. Keep answers concise and clear."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        "Context passages:\n"
                        f"{self._build_context(contexts)}\n\n"
                        "Write a short answer grounded in the context above."
                    ),
                },
            ]
        )
        answer_text = (resp.content or "").strip()

        seen = set()
        citations: List[Dict[str, Any]] = []
        for context in contexts:
            title = context.get("title", "")
            url = context.get("url", "")
            page = context.get("page")
            try:
                page_num = int(page) if page is not None else None
            except (TypeError, ValueError):
                page_num = None

            key = (title, url, page_num)
            if key in seen:
                continue
            seen.add(key)

            citation: Dict[str, Any] = {"title": title, "url": url}
            if page_num is not None:
                citation["page"] = page_num
            citations.append(citation)

        return answer_text, citations[:5]
