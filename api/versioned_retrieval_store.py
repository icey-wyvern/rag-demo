"""PostgreSQL full-text search over one eligible observation per source."""

from datetime import datetime
import math

from sqlalchemy import DateTime, Integer, String, bindparam, text
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

from .evidence import Evidence
from .passage_embedding_provider import EMBEDDING_DIMENSIONS


def eligible_snapshots_sql(as_of: datetime | None) -> str:
    """One local observation per source, shared by every retrieval mode."""
    cutoff = "WHERE ss.observed_at <= :as_of" if as_of is not None else ""
    return f"""
        eligible AS (
            SELECT ss.id, ss.source_id,
                   row_number() OVER (
                       PARTITION BY ss.source_id
                       ORDER BY ss.observed_at DESC, ss.created_at DESC, ss.id DESC
                   ) AS version_rank
            FROM source_snapshots ss
            {cutoff}
        )
    """.strip()


def lexical_statement(as_of: datetime | None):
    # Eligibility is established before the passage join and lexical filter.
    statement = text(f"""
        WITH {eligible_snapshots_sql(as_of)}, search_query AS (
            SELECT websearch_to_tsquery('english', :question) AS tsq
        )
        SELECT s.id AS source_id, s.source_key, s.title AS source_title,
               s.canonical_url, ss.id AS snapshot_id,
               ss.content_sha256 AS snapshot_content_sha256, ss.retrieved_url,
               ss.observed_at, ss.source_published_at, ss.source_updated_at,
               p.id AS passage_id, p.ordinal AS passage_ordinal, p.page,
               p.text, p.text_sha256,
               ts_rank_cd(to_tsvector('english', p.text), search_query.tsq) AS score
        FROM eligible e
        JOIN source_snapshots ss ON ss.id = e.id
        JOIN sources s ON s.id = e.source_id
        JOIN passages p ON p.snapshot_id = e.id
        CROSS JOIN search_query
        WHERE e.version_rank = 1
          AND to_tsvector('english', p.text) @@ search_query.tsq
        ORDER BY score DESC, s.source_key ASC, p.ordinal ASC, p.id ASC
        LIMIT :k
    """).bindparams(
        bindparam("question", type_=String()),
        bindparam("k", type_=Integer()),
    )
    if as_of is not None:
        statement = statement.bindparams(bindparam("as_of", type_=DateTime(timezone=True)))
    return statement


def semantic_statement(as_of: datetime | None):
    # Exact cosine ordering; no ANN index or post-filtered historical vectors.
    statement = text(f"""
        WITH {eligible_snapshots_sql(as_of)}
        SELECT s.id AS source_id, s.source_key, s.title AS source_title,
               s.canonical_url, ss.id AS snapshot_id,
               ss.content_sha256 AS snapshot_content_sha256, ss.retrieved_url,
               ss.observed_at, ss.source_published_at, ss.source_updated_at,
               p.id AS passage_id, p.ordinal AS passage_ordinal, p.page,
               p.text, p.text_sha256,
               pe.embedding <=> :query_vector AS score
        FROM eligible e
        JOIN source_snapshots ss ON ss.id = e.id
        JOIN sources s ON s.id = e.source_id
        JOIN passages p ON p.snapshot_id = e.id
        JOIN passage_embeddings pe ON pe.passage_id = p.id
        WHERE e.version_rank = 1
          AND pe.provider = :provider AND pe.model = :model
        ORDER BY score ASC, s.source_key ASC, p.ordinal ASC, p.id ASC
        LIMIT :k
    """).bindparams(
        bindparam("query_vector", type_=Vector(EMBEDDING_DIMENSIONS)),
        bindparam("provider", type_=String()),
        bindparam("model", type_=String()),
        bindparam("k", type_=Integer()),
    )
    if as_of is not None:
        statement = statement.bindparams(bindparam("as_of", type_=DateTime(timezone=True)))
    return statement


def _evidence(row, rank: int, method: str, score_kind: str) -> Evidence:
    score = float(row["score"])
    if not math.isfinite(score):
        raise ValueError("Retrieval returned a non-finite score")
    return Evidence(
        source_id=row["source_id"], source_key=row["source_key"],
        source_title=row["source_title"], canonical_url=row["canonical_url"],
        snapshot_id=row["snapshot_id"], snapshot_content_sha256=row["snapshot_content_sha256"],
        retrieved_url=row["retrieved_url"], observed_at=row["observed_at"],
        source_published_at=row["source_published_at"], source_updated_at=row["source_updated_at"],
        passage_id=row["passage_id"], passage_ordinal=row["passage_ordinal"], page=row["page"],
        text=row["text"], text_sha256=row["text_sha256"],
        retrieval_method=method, score=score, rank=rank, score_kind=score_kind,
        lexical_rank=rank if method == "postgresql_fts_english" else None,
        semantic_rank=rank if method == "pgvector_cosine_exact" else None,
    )


class VersionedRetrievalStore:
    def __init__(self, session: Session):
        self.session = session

    def begin_consistent_read(self) -> None:
        # Hybrid runs two rankings. Pin them to one PostgreSQL MVCC snapshot
        # so ingestion between statements cannot mix source versions.
        self.session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))

    def search(self, question: str, as_of: datetime | None, k: int) -> list[Evidence]:
        params = {"question": question, "k": k}
        if as_of is not None:
            params["as_of"] = as_of
        rows = self.session.execute(lexical_statement(as_of), params).mappings().all()
        return [_evidence(row, rank, "postgresql_fts_english", "lexical_ts_rank_cd")
                for rank, row in enumerate(rows, 1)]

    def has_embeddings(self, provider: str, model: str) -> bool:
        return bool(self.session.execute(
            text("SELECT EXISTS (SELECT 1 FROM passage_embeddings "
                 "WHERE provider = :provider AND model = :model)"),
            {"provider": provider, "model": model},
        ).scalar_one())

    def semantic_search(
        self, vector: list[float], provider: str, model: str, as_of: datetime | None, k: int,
    ) -> list[Evidence]:
        params = {"query_vector": vector, "provider": provider, "model": model, "k": k}
        if as_of is not None:
            params["as_of"] = as_of
        rows = self.session.execute(semantic_statement(as_of), params).mappings().all()
        return [_evidence(row, rank, "pgvector_cosine_exact", "cosine_distance")
                for rank, row in enumerate(rows, 1)]
