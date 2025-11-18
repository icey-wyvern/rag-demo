from sqlalchemy import Column, Text, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
import uuid

from .db import Base

class Document(Base):
	__tablename__ = "documents"

	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	title = Column(Text, nullable=False)
	base_path = Column(Text, nullable=False)
	url = Column(Text, nullable=False)

	chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
	__tablename__ = "chunks"

	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
	ord = Column(Integer, nullable=False)
	content = Column(Text, nullable=False)
	embedding = Column(Vector(1536))

	document = relationship("Document", back_populates="chunks")
