"""Pydantic models for RAG pipeline per SPEC.md section 8 and 13."""
from typing import Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for an ingested document."""

    document_id: str
    filename: str
    title: str | None = None
    sha256: str
    pages: int
    chunk_count: int
    status: str = "ready"
    uploaded_at: str


class TextChunk(BaseModel):
    """Text chunk with positional metadata."""

    chunk_id: str
    document_id: str
    filename: str
    page: int
    section: str | None = None
    text: str


class Citation(BaseModel):
    """Exact citation structure per SPEC.md section 8."""

    document_id: str
    filename: str
    page: int
    chunk_id: str
    quote: str


class SearchResult(BaseModel):
    """Retrieved and reranked passage with score."""

    chunk_id: str
    document_id: str
    filename: str
    page: int
    section: str | None = None
    text: str
    score: float


class IngestResult(BaseModel):
    """Result of document ingestion."""

    document_id: str
    filename: str
    pages: int
    chunk_count: int
    duration_ms: int
    already_ingested: bool = False
