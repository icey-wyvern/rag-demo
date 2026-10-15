"""Internal composition and serialization for the public versioned query path."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .evidence import Evidence
from .evidence_answer import AnswerResult, EvidenceFirstAnswerService
from .versioned_retriever import RetrievalMode, VersionedRetriever


@dataclass(frozen=True)
class QueryResult:
    evidence: tuple[Evidence, ...]
    answer: AnswerResult | None
    retrieval_mode: RetrievalMode
    as_of: datetime | None
    k: int


class QueryPipeline:
    def __init__(self, retriever: VersionedRetriever, answer_service: EvidenceFirstAnswerService):
        self.retriever = retriever
        self.answer_service = answer_service

    def run(
        self, question: str, *, response_mode: Literal["generate", "chunks"],
        retrieval_mode: RetrievalMode, as_of: datetime | None, k: int,
    ) -> QueryResult:
        evidence = tuple(self.retriever.search(question, mode=retrieval_mode, as_of=as_of, k=k))
        answer = self.answer_service.answer(question, evidence) if response_mode == "generate" else None
        return QueryResult(evidence, answer, retrieval_mode, as_of, k)


def _snippet(text: str, limit: int = 360) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _provenance(item: Evidence) -> dict:
    return {
        "source_key": item.source_key,
        "snapshot_id": str(item.snapshot_id),
        "snapshot_content_sha256": item.snapshot_content_sha256,
        "observed_at": item.observed_at.isoformat(),
        "source_updated_at": item.source_updated_at.isoformat() if item.source_updated_at else None,
        "passage_id": str(item.passage_id),
        "passage_ordinal": item.passage_ordinal,
        "passage_sha256": item.text_sha256,
    }


def serialize_query_result(result: QueryResult) -> dict:
    chunks = [
        {
            "title": item.source_title, "url": item.canonical_url, "page": item.page,
            "snippet": _snippet(item.text), "text": item.text, **_provenance(item),
        }
        for item in result.evidence
    ]
    metadata = {
        "retrieval_mode": result.retrieval_mode, "k": result.k,
        "as_of": result.as_of.isoformat() if result.as_of else None,
        "evidence_count": len(result.evidence),
    }
    if result.answer is None:
        return {"status": "retrieved", "answer": "", "citations": [], "sources": [],
                "claims": [], "chunks": chunks, "retrieval": metadata}
    if result.answer.status == "abstained":
        return {"status": "abstained", "answer": result.answer.abstention_reason or "Insufficient evidence.",
                "abstention_reason": result.answer.abstention_reason,
                "citations": [], "sources": [], "claims": [], "chunks": chunks, "retrieval": metadata}
    links = []
    claims = []
    seen_passages = set()
    for claim in result.answer.claims:
        cited = []
        for citation in claim.citations:
            item = citation.evidence
            cited.append({
                "evidence_id": citation.evidence_id,
                "supporting_quote": citation.supporting_quote,
                "title": item.source_title, "url": item.canonical_url, "page": item.page,
                **_provenance(item),
            })
            key = (item.snapshot_id, item.passage_id)
            if key not in seen_passages:
                links.append({"title": item.source_title, "url": item.canonical_url,
                              "page": item.page, **_provenance(item)})
                seen_passages.add(key)
        claims.append({"text": claim.text, "citations": cited})
    return {"status": "answered", "answer": result.answer.answer_text,
            "citations": links, "sources": links, "claims": claims,
            "chunks": chunks, "retrieval": metadata}
