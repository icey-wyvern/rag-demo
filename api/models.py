import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .db import Base


class CorpusState(Base):
    __tablename__ = "corpus_state"

    id = Column(Integer, primary_key=True, default=1)
    status = Column(Text, nullable=False, default="pending")
    ingest_version = Column(Integer, nullable=False, default=1)
    documents_total = Column(Integer, nullable=False, default=0)
    documents_ready = Column(Integer, nullable=False, default=0)
    last_error = Column(Text, nullable=True)


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    base_path = Column(Text, nullable=False, unique=True)
    url = Column(Text, nullable=False)
    source_sha256 = Column(Text, nullable=True)
    ingest_version = Column(Integer, nullable=False, default=1)
    chunk_count = Column(Integer, nullable=False, default=0)
    status = Column(Text, nullable=False, default="pending")

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("doc_id", "ord", name="uq_chunks_doc_ord"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    ord = Column(Integer, nullable=False)
    page = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)

    document = relationship("Document", back_populates="chunks")
