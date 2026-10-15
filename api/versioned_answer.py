"""Internal composition of versioned retrieval and evidence-first generation."""

from datetime import datetime

from .evidence_answer import AnswerResult, EvidenceFirstAnswerService
from .versioned_retriever import DEFAULT_K, RetrievalMode, VersionedRetriever


class VersionedAnswerService:
    def __init__(self, retriever: VersionedRetriever, answer_service: EvidenceFirstAnswerService):
        self.retriever = retriever
        self.answer_service = answer_service

    def answer(
        self, question: str, *, mode: RetrievalMode = "lexical",
        as_of: datetime | None = None, k: int = DEFAULT_K,
    ) -> AnswerResult:
        evidence = self.retriever.search(question, mode=mode, as_of=as_of, k=k)
        return self.answer_service.answer(question, evidence)
