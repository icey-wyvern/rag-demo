"""Small versioned-corpus readiness view for the existing status UI."""

from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from .versioned_retrieval_store import eligible_snapshots_sql


def versioned_corpus_state(
    session, embedding_model: str, ingest_version: int, *, provider_configured: bool = True,
) -> dict:
    try:
        rows = session.execute(text(f"""
        WITH {eligible_snapshots_sql(None)}
        SELECT
          (SELECT count(*) FROM sources) AS source_count,
          (SELECT count(*) FROM source_snapshots) AS snapshot_count,
          (SELECT count(*) FROM passages p JOIN eligible e ON e.id = p.snapshot_id
           WHERE e.version_rank = 1) AS current_passages,
          (SELECT count(*) FROM passage_embeddings pe
           JOIN passages p ON p.id = pe.passage_id
           JOIN eligible e ON e.id = p.snapshot_id
           WHERE e.version_rank = 1 AND pe.provider = 'openai' AND pe.model = :model) AS embedded_passages,
          (SELECT count(DISTINCT e.source_id) FROM passage_embeddings pe
           JOIN passages p ON p.id = pe.passage_id
           JOIN eligible e ON e.id = p.snapshot_id
           WHERE e.version_rank = 1 AND pe.provider = 'openai' AND pe.model = :model) AS embedded_sources
        """), {"model": embedding_model}).mappings().one()
    except ProgrammingError as exc:
        if getattr(exc.orig, "sqlstate", None) != "42P01":
            raise
        return {
            "status": "pending", "ingest_version": ingest_version,
            "documents_total": 0, "documents_ready": 0,
            "last_error": None, "is_ready": False,
            "versioned": {
                "schema_ready": False, "sources": 0, "snapshots": 0,
                "current_passages": 0, "embedded_passages": 0,
                "lexical_ready": False, "semantic_ready": False,
                "provider_configured": provider_configured,
            },
        }
    source_count = int(rows["source_count"])
    snapshot_count = int(rows["snapshot_count"])
    passages = int(rows["current_passages"])
    embedded_passages = int(rows["embedded_passages"])
    embedded_sources = int(rows["embedded_sources"])
    lexical_ready = passages > 0
    semantic_ready = embedded_passages > 0
    ready = lexical_ready and semantic_ready and provider_configured
    return {
        "status": "ready" if ready else "error" if lexical_ready and semantic_ready and not provider_configured
                  else "running" if lexical_ready else "pending",
        "ingest_version": ingest_version,
        "documents_total": source_count,
        "documents_ready": embedded_sources,
        "last_error": "OpenAI provider is not configured for default hybrid queries." if lexical_ready and semantic_ready and not provider_configured else None,
        "is_ready": ready,
        "versioned": {
            "schema_ready": True, "sources": source_count, "snapshots": snapshot_count,
            "current_passages": passages, "embedded_passages": embedded_passages,
            "lexical_ready": lexical_ready, "semantic_ready": semantic_ready,
            "provider_configured": provider_configured,
        },
    }
