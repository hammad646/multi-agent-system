"""PDF Ingestion pipeline per SPEC.md section 13.

Validates PDF -> sha256 -> PyMuPDF (pypdf fallback) -> chunk (~800-1000 chars, 150 overlap)
-> Pinecone embeddings & upsert -> SQLite storage (DocumentRepository) -> registry row ready.
"""
import argparse
import hashlib
from io import BytesIO
from pathlib import Path
import re
import sys
import time
from typing import Any
import uuid

from app.config import settings
from app.db.base import get_session_factory, init_db
from app.db.models import DocumentModel, ChunkModel
from app.db.repositories import DocumentRepository
from app.rag.models import IngestResult, TextChunk

_global_pinecone_client = None


def get_pinecone_client() -> Any:
    """Get Pinecone client or FakePinecone when in MOCK_MODE or no API key."""
    global _global_pinecone_client
    if _global_pinecone_client is not None:
        return _global_pinecone_client

    if settings.MOCK_MODE or not settings.PINECONE_API_KEY:
        from app.testing.fakes import FakePinecone
        _global_pinecone_client = FakePinecone(api_key="mock-key")
        return _global_pinecone_client

    try:
        from pinecone import Pinecone
        _global_pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        return _global_pinecone_client
    except Exception:
        from app.testing.fakes import FakePinecone
        _global_pinecone_client = FakePinecone(api_key="mock-key")
        return _global_pinecone_client


def set_pinecone_client(client: Any) -> None:
    """Explicitly set Pinecone client (used in tests)."""
    global _global_pinecone_client
    _global_pinecone_client = client


def compute_sha256(data: bytes) -> str:
    """Compute sha256 hex digest of file bytes."""
    return hashlib.sha256(data).hexdigest()


def extract_pages(file_bytes: bytes, filename: str) -> list[tuple[int, str, str | None]]:
    """Extract pages as (page_number, text, section) using PyMuPDF with pypdf fallback."""
    pages_data: list[tuple[int, str, str | None]] = []

    # 1. Try PyMuPDF
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for idx, page in enumerate(doc):
            page_num = idx + 1
            text = page.get_text()
            section = None
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines[:3]:
                if len(line) < 60 and (line.isupper() or line.istitle()):
                    section = line
                    break
            pages_data.append((page_num, text, section))
        return pages_data
    except Exception:
        pass

    # 2. Fallback to pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(BytesIO(file_bytes))
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            text = page.extract_text() or ""
            section = None
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines[:3]:
                if len(line) < 60 and (line.isupper() or line.istitle()):
                    section = line
                    break
            pages_data.append((page_num, text, section))
        return pages_data
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF {filename}: {exc}") from exc


def chunk_text(
    pages: list[tuple[int, str, str | None]],
    document_id: str,
    filename: str,
    target_size: int = 900,
    overlap: int = 150,
) -> list[TextChunk]:
    """Split pages into overlapping chunks (~800-1000 chars, 150 overlap)."""
    chunks: list[TextChunk] = []

    for page_num, text, section in pages:
        clean_text = re.sub(r"\s+", " ", text).strip()
        if not clean_text:
            continue

        start = 0
        chunk_idx = 0
        text_len = len(clean_text)

        while start < text_len:
            end = min(start + target_size, text_len)
            # Try to break at a sentence or word boundary if not at end
            if end < text_len:
                boundary = clean_text.rfind(". ", start + target_size // 2, end)
                if boundary != -1:
                    end = boundary + 1
                else:
                    space_boundary = clean_text.rfind(" ", start + target_size // 2, end)
                    if space_boundary != -1:
                        end = space_boundary

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                chunk_id = f"{document_id}_p{page_num}_{chunk_idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        page=page_num,
                        section=section,
                        text=chunk_content,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break
            start = end - overlap

    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    repo: DocumentRepository,
    pinecone_client: Any | None = None,
) -> IngestResult:
    """Ingest a PDF file into the database and Pinecone."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path.name}")

    file_bytes = path.read_bytes()
    if not file_bytes:
        raise ValueError(f"PDF file is empty: {path.name}")

    start_time = time.perf_counter()
    sha256 = compute_sha256(file_bytes)

    # 1. Check for duplicates (skip if already ingested)
    existing = repo.get_document_by_sha256(sha256)
    if existing:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return IngestResult(
            document_id=existing.document_id,
            filename=existing.filename,
            pages=existing.pages,
            chunk_count=existing.chunk_count,
            duration_ms=duration_ms,
            already_ingested=True,
        )

    # 2. Parse PDF pages
    pages = extract_pages(file_bytes, path.name)
    document_id = str(uuid.uuid4())
    total_pages = len(pages)

    # 3. Chunk text
    chunks = chunk_text(pages, document_id, path.name)

    # 4. Generate embeddings and upsert to Pinecone
    pc = pinecone_client or get_pinecone_client()
    index = pc.Index(settings.PINECONE_INDEX)

    if chunks:
        chunk_texts = [c.text for c in chunks]
        embeddings_resp = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=chunk_texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )

        vectors = []
        for i, c in enumerate(chunks):
            values = embeddings_resp.data[i].values
            vectors.append(
                (
                    c.chunk_id,
                    values,
                    {
                        "document_id": c.document_id,
                        "filename": c.filename,
                        "page": c.page,
                        "chunk_id": c.chunk_id,
                        "section": c.section or "",
                        "text": c.text,
                    },
                )
            )

        # Upsert with namespace = document_id
        index.upsert(vectors=vectors, namespace=document_id)

    # 5. Store document and chunks in SQLite DB
    doc_model = DocumentModel(
        document_id=document_id,
        filename=path.name,
        title=path.stem,
        sha256=sha256,
        pages=total_pages,
        chunk_count=len(chunks),
        status="ready",
    )
    chunk_models = [
        ChunkModel(
            chunk_id=c.chunk_id,
            document_id=document_id,
            page=c.page,
            section=c.section,
            text=c.text,
        )
        for c in chunks
    ]
    repo.add_document(doc_model, chunk_models)

    duration_ms = int((time.perf_counter() - start_time) * 1000)
    return IngestResult(
        document_id=document_id,
        filename=path.name,
        pages=total_pages,
        chunk_count=len(chunks),
        duration_ms=duration_ms,
        already_ingested=False,
    )


def list_documents(repo: DocumentRepository) -> list[DocumentModel]:
    """Return all ingested documents."""
    return repo.list_documents()


def delete_document(
    document_id: str,
    repo: DocumentRepository,
    pinecone_client: Any | None = None,
) -> bool:
    """Delete document from database and Pinecone namespace."""
    pc = pinecone_client or get_pinecone_client()
    index = pc.Index(settings.PINECONE_INDEX)
    # Remove from Pinecone
    try:
        index.delete(delete_all=True, namespace=document_id)
    except Exception:
        pass

    # Remove from database (cascades to chunks)
    return repo.delete_document(document_id)


def main() -> None:
    """CLI entrypoint for ingestion, listing, and deletion."""
    parser = argparse.ArgumentParser(description="PDF Ingestion and Document Management")
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF file to ingest")
    parser.add_argument("--list", action="store_true", help="List all ingested documents")
    parser.add_argument("--delete", metavar="ID", help="Delete ingested document by ID")
    args = parser.parse_args()

    init_db()
    session_factory = get_session_factory()
    session = session_factory()
    repo = DocumentRepository(session)

    try:
        if args.list:
            docs = list_documents(repo)
            if not docs:
                print("No documents ingested yet.")
                return
            print(f"{'ID':<38} {'Filename':<30} {'Pages':<6} {'Chunks':<8} {'Status':<10}")
            print("-" * 95)
            for d in docs:
                print(f"{d.document_id:<38} {d.filename:<30} {d.pages:<6} {d.chunk_count:<8} {d.status:<10}")
            return

        if args.delete:
            success = delete_document(args.delete, repo)
            if success:
                print(f"Document {args.delete} deleted successfully.")
            else:
                print(f"Document {args.delete} not found.")
            return

        if args.pdf_path:
            res = ingest_pdf(args.pdf_path, repo)
            if res.already_ingested:
                print(f"Already ingested: {res.filename} (id: {res.document_id}, {res.pages} pages, {res.chunk_count} chunks)")
            else:
                print(f"Successfully ingested {res.filename}:")
                print(f"  ID: {res.document_id}")
                print(f"  Pages: {res.pages}")
                print(f"  Chunks: {res.chunk_count}")
                print(f"  Time: {res.duration_ms} ms")
            return

        parser.print_help()
    finally:
        session.close()


if __name__ == "__main__":
    main()
