"""Additive provenance models; the current RAG path still uses Document and Chunk."""

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, ForeignKeyConstraint, Index, Integer, String, Text, UniqueConstraint, func, literal_column
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .db import Base


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("source_key", name="uq_sources_source_key"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_key = Column(Text, nullable=False)
    canonical_url = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    snapshots = relationship("SourceSnapshot", back_populates="source", foreign_keys="SourceSnapshot.source_id")


class SourceSnapshot(Base):
    """One observed content version; rows are intended to be append-only."""

    __tablename__ = "source_snapshots"
    __table_args__ = (
        UniqueConstraint("source_id", "content_sha256", name="uq_source_snapshots_source_hash"),
        UniqueConstraint("source_id", "id", name="uq_source_snapshots_source_id_id"),
        ForeignKeyConstraint(
            ["source_id", "predecessor_snapshot_id"],
            ["source_snapshots.source_id", "source_snapshots.id"],
            name="fk_source_snapshots_predecessor_same_source",
            ondelete="RESTRICT",
        ),
        Index("ix_source_snapshots_source_observed_at", "source_id", "observed_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("sources.id", name="fk_source_snapshots_source", ondelete="RESTRICT"), nullable=False)
    retrieved_url = Column(Text, nullable=False)
    content_sha256 = Column(String(64), nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    source_published_at = Column(DateTime(timezone=True), nullable=True)
    source_updated_at = Column(DateTime(timezone=True), nullable=True)
    effective_from = Column(DateTime(timezone=True), nullable=True)
    predecessor_snapshot_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    source = relationship("Source", back_populates="snapshots", foreign_keys=[source_id])
    predecessor = relationship(
        "SourceSnapshot",
        remote_side=[id],
        foreign_keys=[source_id, predecessor_snapshot_id],
        viewonly=True,
    )
    passages = relationship("Passage", back_populates="snapshot")


class Passage(Base):
    __tablename__ = "passages"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "ordinal", name="uq_passages_snapshot_ordinal"),
        Index(
            "ix_passages_text_english_fts",
            func.to_tsvector(literal_column("'english'"), literal_column("text")),
            postgresql_using="gin",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_id = Column(UUID(as_uuid=True), ForeignKey("source_snapshots.id", name="fk_passages_snapshot", ondelete="RESTRICT"), nullable=False)
    ordinal = Column(Integer, nullable=False)
    page = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    text_sha256 = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    snapshot = relationship("SourceSnapshot", back_populates="passages")
    embeddings = relationship("PassageEmbedding", back_populates="passage", passive_deletes="all")


class PassageEmbedding(Base):
    """Derived vector for one immutable passage and one provider/model identity."""

    __tablename__ = "passage_embeddings"
    __table_args__ = (
        UniqueConstraint("passage_id", "provider", "model", name="uq_passage_embeddings_identity"),
        CheckConstraint("dimensions = 1536", name="ck_passage_embeddings_dimensions_1536"),
        Index("ix_passage_embeddings_provider_model", "provider", "model"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    passage_id = Column(
        UUID(as_uuid=True), ForeignKey("passages.id", name="fk_passage_embeddings_passage", ondelete="CASCADE"),
        nullable=False,
    )
    provider = Column(Text, nullable=False)
    model = Column(Text, nullable=False)
    dimensions = Column(Integer, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    passage = relationship("Passage", back_populates="embeddings")
