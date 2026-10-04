"""SQLAlchemy models for documents, chunks, scheduled emails, and workflows.

PostgreSQL-compatible types: UUID string IDs, UTC timestamps, standard columns.
"""
from datetime import datetime, timezone
import uuid
from typing import Any
from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    TypeDecorator,
)
from sqlalchemy.orm import relationship
from app.db.base import Base


class UTCDateTime(TypeDecorator):
    """DateTime type that enforces timezone-aware UTC datetime.

    Rejects naive datetimes per SPEC guardrail 7.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is not None:
            if not isinstance(value, datetime):
                raise TypeError(f"Expected datetime object, got {type(value)}")
            if value.tzinfo is None:
                raise ValueError("Naive datetimes are rejected per SPEC.md guardrails.")
            return value.astimezone(timezone.utc)
        return value

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is not None:
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        return value


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Generate string UUID4."""
    return str(uuid.uuid4())


class DocumentModel(Base):
    """Document metadata table."""

    __tablename__ = "documents"

    document_id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    title = Column(String(255), nullable=True)
    sha256 = Column(String(64), nullable=False, unique=True, index=True)
    pages = Column(Integer, nullable=False, default=1)
    chunk_count = Column(Integer, nullable=False, default=0)
    status = Column(String(32), nullable=False, default="ready")
    uploaded_at = Column(UTCDateTime, nullable=False, default=utc_now)

    chunks = relationship(
        "ChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkModel(Base):
    """Document text chunks table for BM25 search."""

    __tablename__ = "chunks"

    chunk_id = Column(String(64), primary_key=True)
    document_id = Column(
        String(36),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page = Column(Integer, nullable=False)
    section = Column(String(255), nullable=True)
    text = Column(Text, nullable=False)

    document = relationship("DocumentModel", back_populates="chunks")


class ScheduledEmailModel(Base):
    """Scheduled emails persisted for background worker dispatch."""

    __tablename__ = "scheduled_emails"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(64), nullable=False, default="default_user")
    recipients = Column(String(1024), nullable=False)
    cc = Column(String(1024), nullable=True)
    bcc = Column(String(1024), nullable=True)
    subject = Column(String(512), nullable=False)
    body = Column(Text, nullable=False)
    scheduled_at = Column(UTCDateTime, nullable=False)
    timezone = Column(String(64), nullable=False, default="UTC")
    status = Column(String(32), nullable=False, default="pending", index=True)
    created_at = Column(UTCDateTime, nullable=False, default=utc_now)
    approved_at = Column(UTCDateTime, nullable=True)
    sent_at = Column(UTCDateTime, nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    last_error = Column(Text, nullable=True)
    gmail_message_id = Column(String(128), nullable=True)


class WorkflowModel(Base):
    """Workflow execution run records."""

    __tablename__ = "workflows"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    workflow_type = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False, default="pending", index=True)
    steps = Column(Text, nullable=False, default="{}")
    created_at = Column(UTCDateTime, nullable=False, default=utc_now)
    updated_at = Column(UTCDateTime, nullable=False, default=utc_now)
