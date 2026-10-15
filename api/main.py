"""Public API backed by the versioned evidence-first query pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import get_settings
from .db import SessionLocal
from .evidence_answer import AnswerError, AnswerProviderError, EvidenceFirstAnswerService
from .openai_answer_provider import OpenAIAnswerProvider
from .passage_embedding_provider import OpenAIPassageEmbeddingProvider
from .query_service import QueryPipeline, serialize_query_result
from .versioned_readiness import versioned_corpus_state
from .versioned_retriever import DEFAULT_K, MAX_K, SemanticRetrievalError, VersionedRetriever


app = FastAPI(title="gvt-rag")
static_dir = Path(__file__).resolve().parents[1] / "static"


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str | None = None  # Existing frontend field.
    question: str | None = None
    mode: Literal["chunks", "generate"] = "generate"
    retrieval_mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    as_of: datetime | None = None
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K, strict=True)

    @field_validator("as_of", mode="before")
    @classmethod
    def as_of_is_timestamp(cls, value):
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError("as_of must be a timezone-aware timestamp")
        return value

    @field_validator("as_of")
    @classmethod
    def as_of_is_aware(cls, value):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("as_of must be timezone-aware")
        return value

    @model_validator(mode="after")
    def has_question(self):
        q = self.q.strip() if isinstance(self.q, str) else ""
        question = self.question.strip() if isinstance(self.question, str) else ""
        if not q and not question:
            raise ValueError("question must be non-empty")
        if q and question and q != question:
            raise ValueError("q and question must agree when both are supplied")
        return self

    @property
    def query_text(self) -> str:
        return (self.q or self.question or "").strip()


class _ConfiguredAnswerProvider:
    """Construct the network adapter only if the answer service calls generate."""

    def generate(self, question: str, formatted_evidence: str, instructions: str) -> object:
        return OpenAIAnswerProvider.from_settings().generate(question, formatted_evidence, instructions)


def get_query_pipeline() -> QueryPipeline:
    settings = get_settings()
    embedding_provider = (
        OpenAIPassageEmbeddingProvider(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)
        if settings.OPENAI_API_KEY else None
    )
    retriever = VersionedRetriever(SessionLocal, embedding_provider=embedding_provider)
    return QueryPipeline(retriever, EvidenceFirstAnswerService(_ConfiguredAnswerProvider()))


def get_versioned_status() -> dict:
    settings = get_settings()
    try:
        with SessionLocal() as session:
            return versioned_corpus_state(
                session, settings.EMBED_MODEL, settings.CORPUS_VERSION,
                provider_configured=bool(settings.OPENAI_API_KEY),
            )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Versioned corpus status unavailable.") from exc


@app.get("/health")
def health(corpus: dict = Depends(get_versioned_status)):
    return {"ok": True, "corpus": corpus}


@app.get("/api/status")
def corpus_status(corpus: dict = Depends(get_versioned_status)):
    return corpus


@app.post("/api/query")
def query(
    req: QueryRequest,
    corpus: dict = Depends(get_versioned_status),
    pipeline: QueryPipeline = Depends(get_query_pipeline),
):
    readiness = corpus.get("versioned", {})
    if not readiness.get("lexical_ready"):
        raise HTTPException(status_code=503, detail="Versioned corpus is not available yet.")
    if req.retrieval_mode in ("semantic", "hybrid") and not readiness.get("semantic_ready"):
        raise HTTPException(status_code=503, detail="Semantic index unavailable for requested retrieval mode.")
    if req.retrieval_mode in ("semantic", "hybrid") and not readiness.get("provider_configured", True):
        raise HTTPException(status_code=503, detail="Semantic retrieval provider is not configured.")
    try:
        result = pipeline.run(
            req.query_text, response_mode=req.mode, retrieval_mode=req.retrieval_mode,
            as_of=req.as_of, k=req.k,
        )
        return serialize_query_result(result)
    except SemanticRetrievalError as exc:
        raise HTTPException(status_code=503, detail="Semantic index unavailable for requested retrieval mode.") from exc
    except AnswerProviderError as exc:
        raise HTTPException(status_code=503, detail="Answer generation unavailable.") from exc
    except AnswerError as exc:
        raise HTTPException(status_code=502, detail="Generated answer failed evidence validation.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid query options.") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Versioned query unavailable.") from exc


if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
else:
    @app.get("/")
    def root():
        return {"name": "gvt-rag", "ui_ready": False,
                "message": "Frontend assets have not been built yet."}
