"""Opt-in OpenAI structured-output adapter for the internal answer path."""

import json

from .evidence_answer import AnswerProviderError, StructuredAnswer


class OpenAIAnswerProvider:
    def __init__(self, *, model: str, api_key: str, timeout_seconds: float = 30.0):
        if not model.strip() or not api_key.strip() or timeout_seconds <= 0:
            raise ValueError("OpenAI generation model, API key, and positive timeout are required")
        self.model = model
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client = None

    @classmethod
    def from_settings(cls) -> "OpenAIAnswerProvider":
        from .config import get_settings, validate_generation_settings

        settings = get_settings()
        validate_generation_settings(settings)
        return cls(model=settings.GEN_MODEL, api_key=settings.OPENAI_API_KEY)

    def generate(self, question: str, formatted_evidence: str, instructions: str) -> StructuredAnswer:
        from langchain_openai import ChatOpenAI

        try:
            if self._client is None:
                self._client = ChatOpenAI(
                    model=self.model,
                    openai_api_key=self._api_key,
                    temperature=0,
                    timeout=self.timeout_seconds,
                    max_retries=2,
                ).with_structured_output(StructuredAnswer, method="json_schema", strict=True)
            payload = json.dumps({
                "question": question,
                "evidence": json.loads(formatted_evidence),
            }, ensure_ascii=False, sort_keys=True)
            response = self._client.invoke([
                {"role": "system", "content": instructions},
                {"role": "user", "content": payload},
            ])
        except Exception as exc:
            raise AnswerProviderError("OpenAI structured generation failed") from exc
        if not isinstance(response, StructuredAnswer):
            raise AnswerProviderError("OpenAI returned an invalid structured response")
        return response
