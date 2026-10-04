"""Hybrid retrieval pipeline (Dense + BM25 with Reciprocal Rank Fusion).

All database operations strictly access SQLite via DocumentRepository.
Combines dense search and BM25 sparse search via RRF, followed by reranking.
"""
import re
from typing import Any
from rank_bm25 import BM25Okapi

from app.config import settings
from app.db.repositories import DocumentRepository
from app.rag.ingest import get_pinecone_client
from app.rag.models import SearchResult
from app.rag.rerank import rerank_results, DEFAULT_RERANK_THRESHOLD


def resolve_document_ids(
    query: str,
    repo: DocumentRepository,
    explicit_document_ids: list[str] | None = None,
) -> list[str] | None:
    """Resolve document IDs from explicit filter or document name mentions in query."""
    if explicit_document_ids:
        return explicit_document_ids

    all_docs = repo.list_documents()
    if not all_docs:
        return None

    query_lower = query.lower()
    matched_ids: list[str] = []

    for doc in all_docs:
        # Check exact filename
        if doc.filename.lower() in query_lower:
            matched_ids.append(doc.document_id)
            continue
        # Check title / stem
        if doc.title and len(doc.title) >= 4 and doc.title.lower() in query_lower:
            matched_ids.append(doc.document_id)
            continue
        # Check significant words in filename (e.g., "policy" in "company_policy.pdf")
        parts = [p.lower() for p in re.split(r"[-_\s.]", doc.filename) if len(p) >= 4]
        if any(p in query_lower for p in parts):
            matched_ids.append(doc.document_id)

    return matched_ids if matched_ids else None


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


from tenacity import retry, stop_after_attempt, wait_exponential
from app.logging import get_logger

logger = get_logger("rag.retrieve")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=1.0),
    reraise=True,
)
def _call_pinecone_embed(pc: Any, query: str) -> list[float]:
    """Execute Pinecone embedding generation with retry and exponential backoff."""
    embed_resp = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"},
    )
    return embed_resp.data[0].values


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=1.0),
    reraise=True,
)
def _call_pinecone_query(index: Any, **kwargs: Any) -> dict[str, Any]:
    """Execute Pinecone index query with retry and exponential backoff."""
    return index.query(**kwargs)


def dense_search(
    query: str,
    top_k: int = 20,
    target_document_ids: list[str] | None = None,
    pinecone_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Dense vector search across Pinecone namespaces with target filtering and retry."""
    pc = pinecone_client or get_pinecone_client()
    index = pc.Index(settings.PINECONE_INDEX)

    try:
        query_vector = _call_pinecone_embed(pc, query)
    except Exception as exc:
        logger.error("pinecone_embed_failed", error=str(exc))
        return []

    results: list[dict[str, Any]] = []

    # If target document IDs are given, query those namespaces
    if target_document_ids:
        for doc_id in target_document_ids:
            try:
                resp = _call_pinecone_query(
                    index,
                    vector=query_vector,
                    top_k=top_k,
                    namespace=doc_id,
                    include_metadata=True,
                )
            except Exception as exc:
                logger.error("pinecone_query_failed", namespace=doc_id, error=str(exc))
                resp = {"matches": []}

            for m in resp.get("matches", []):
                meta = m.get("metadata", {})
                if meta:
                    results.append(
                        {
                            "chunk_id": meta.get("chunk_id", m["id"]),
                            "document_id": meta.get("document_id", doc_id),
                            "filename": meta.get("filename", ""),
                            "page": int(meta.get("page", 1)),
                            "section": meta.get("section"),
                            "text": meta.get("text", ""),
                            "score": float(m.get("score", 0.0)),
                        }
                    )
    else:
        # Cross-namespace search
        try:
            resp = _call_pinecone_query(
                index,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )
        except Exception as exc:
            logger.error("pinecone_query_failed", error=str(exc))
            resp = {"matches": []}

        for m in resp.get("matches", []):
            meta = m.get("metadata", {})
            if meta:
                results.append(
                    {
                        "chunk_id": meta.get("chunk_id", m["id"]),

                        "document_id": meta.get("document_id", ""),
                        "filename": meta.get("filename", ""),
                        "page": int(meta.get("page", 1)),
                        "section": meta.get("section"),
                        "text": meta.get("text", ""),
                        "score": float(m.get("score", 0.0)),
                    }
                )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def bm25_search(
    query: str,
    repo: DocumentRepository,
    top_k: int = 20,
    target_document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """BM25 sparse search over database chunks retrieved strictly via DocumentRepository."""
    chunks = repo.get_all_chunks(filter_doc_ids=target_document_ids)
    if not chunks:
        return []

    # Map document_id -> filename
    docs = {d.document_id: d.filename for d in repo.list_documents()}

    corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    scored_chunks = []
    query_set = set(query_tokens)
    for idx, score in enumerate(scores):
        chunk_tokens = corpus[idx]
        overlap = query_set.intersection(chunk_tokens)
        if score > 0 or overlap:
            effective_score = float(score) if score > 0 else (len(overlap) / max(len(query_set), 1))
            c = chunks[idx]
            scored_chunks.append(
                (
                    effective_score,
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "filename": docs.get(c.document_id, "unknown.pdf"),
                        "page": c.page,
                        "section": c.section,
                        "text": c.text,
                        "score": effective_score,
                    },
                )
            )

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_chunks[:top_k]]


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Fuse dense and sparse rankings using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(chunk) = sum(1.0 / (k + rank))
    Ensures dense and BM25 are fused rather than exposed independently.
    """
    chunk_map: dict[str, dict[str, Any]] = {}
    rrf_scores: dict[str, float] = {}

    # Rank 1-indexed
    for rank, item in enumerate(dense_results, 1):
        cid = item["chunk_id"]
        chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        if cid not in chunk_map:
            chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    fused = []
    for cid, score in rrf_scores.items():
        candidate = dict(chunk_map[cid])
        candidate["rrf_score"] = score
        fused.append(candidate)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused[:top_n]


def hybrid_search(
    query: str,
    repo: DocumentRepository,
    top_k: int = 5,
    document_ids: list[str] | None = None,
    threshold: float = DEFAULT_RERANK_THRESHOLD,
    pinecone_client: Any | None = None,
) -> tuple[list[SearchResult], bool]:
    """Execute end-to-end hybrid retrieval: filter resolution -> Dense + BM25 -> RRF -> Rerank."""
    target_docs = resolve_document_ids(query, repo, explicit_document_ids=document_ids)
    if not target_docs:
        all_docs = repo.list_documents()
        if all_docs:
            target_docs = [d.document_id for d in all_docs]

    # Run dense and sparse searches
    dense_hits = dense_search(
        query=query,
        top_k=20,
        target_document_ids=target_docs,
        pinecone_client=pinecone_client,
    )
    bm25_hits = bm25_search(
        query=query,
        repo=repo,
        top_k=20,
        target_document_ids=target_docs,
    )

    # Fuse via RRF
    fused_candidates = reciprocal_rank_fusion(dense_hits, bm25_hits, top_n=20)
    if not fused_candidates:
        return [], True

    # Rerank with bge-reranker-v2-m3 and apply score threshold
    return rerank_results(
        query=query,
        candidates=fused_candidates,
        top_n=top_k,
        threshold=threshold,
        pinecone_client=pinecone_client,
    )
