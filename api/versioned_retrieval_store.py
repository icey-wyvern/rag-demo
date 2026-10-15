"""PostgreSQL full-text search over one eligible observation per source."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, bindparam, text
from sqlalchemy.orm import Session

from .evidence import Evidence


def lexical_statement(as_of: datetime | None):
    # The window is applied before the passage join and lexical filter. A prior
    # version cannot fill a hit that is absent from the latest eligible version.
    cutoff = "WHERE ss.observed_at <= :as_of" if as_of is not None else ""
    statement = text(f"""
        WITH eligible AS (
            SELECT ss.id, ss.source_id,
                   row_number() OVER (
                       PARTITION BY ss.source_id
                       ORDER BY ss.observed_at DESC, ss.created_at DESC, ss.id DESC
                   ) AS version_rank
            FROM source_snapshots ss
            {cutoff}
        ), search_query AS (
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


class VersionedRetrievalStore:
    def __init__(self, session: Session):
        self.session = session

    def search(self, question: str, as_of: datetime | None, k: int) -> list[Evidence]:
        params = {"question": question, "k": k}
        if as_of is not None:
            params["as_of"] = as_of
        rows = self.session.execute(lexical_statement(as_of), params).mappings().all()
        return [
            Evidence(
                source_id=row["source_id"],
                source_key=row["source_key"],
                source_title=row["source_title"],
                canonical_url=row["canonical_url"],
                snapshot_id=row["snapshot_id"],
                snapshot_content_sha256=row["snapshot_content_sha256"],
                retrieved_url=row["retrieved_url"],
                observed_at=row["observed_at"],
                source_published_at=row["source_published_at"],
                source_updated_at=row["source_updated_at"],
                passage_id=row["passage_id"],
                passage_ordinal=row["passage_ordinal"],
                page=row["page"],
                text=row["text"],
                text_sha256=row["text_sha256"],
                retrieval_method="postgresql_fts_english",
                score=float(row["score"]),
                rank=rank,
            )
            for rank, row in enumerate(rows, 1)
        ]
