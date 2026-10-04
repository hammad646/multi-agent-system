from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
from app.logging import get_logger
from app.rag.ingest import get_pinecone_client
from app.rag.models import SearchResult



logger = get_logger("rag.rerank")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=1.0),
    reraise=True,
)
def _call_pinecone_rerank(pc: Any, **kwargs: Any) -> Any:
    """Execute Pinecone rerank call with exponential backoff."""
    return pc.inference.rerank(**kwargs)


# Minimum reranker threshold below which content is deemed "not found"
DEFAULT_RERANK_THRESHOLD = 0.15



def rerank_results(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int = 5,
    threshold: float = DEFAULT_RERANK_THRESHOLD,
    pinecone_client: Any | None = None,
) -> tuple[list[SearchResult], bool]:
    """Rerank candidates using Pinecone bge-reranker-v2-m3.

    Returns:
        (reranked_search_results, is_below_threshold)
    """
    if not candidates:
        return [], True

    pc = pinecone_client or get_pinecone_client()

    documents_to_rerank = [
        {"id": c["chunk_id"], "text": c["text"]} for c in candidates
    ]

    try:
        rerank_resp = _call_pinecone_rerank(
            pc,
            model="bge-reranker-v2-m3",
            query=query,
            documents=documents_to_rerank,
            top_n=min(top_n, len(candidates)),
            return_documents=True,
        )
    except Exception as exc:
        logger.error("pinecone_rerank_failed", error=str(exc))
        rerank_resp = None


    results: list[SearchResult] = []
    if rerank_resp and hasattr(rerank_resp, "data"):
        for item in rerank_resp.data:
            orig_idx = getattr(item, "index", 0)
            score = float(getattr(item, "score", 0.0))
            cand = candidates[orig_idx]
            results.append(
                SearchResult(
                    chunk_id=cand["chunk_id"],
                    document_id=cand["document_id"],
                    filename=cand["filename"],
                    page=cand["page"],
                    section=cand.get("section"),
                    text=cand["text"],
                    score=score,
                )
            )
    else:
        # Fallback: keep existing order
        for c in candidates[:top_n]:
            results.append(
                SearchResult(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    filename=c["filename"],
                    page=c["page"],
                    section=c.get("section"),
                    text=c["text"],
                    score=float(c.get("score", 0.5)),
                )
            )

    if not results or results[0].score < threshold:
        return results, True

    return results, False
