from typing import List, Optional
import logging

from app.embeddings.embedding_client import get_embedding_client
from app.vectorstore.milvus_client import get_milvus_client
from app.models.schemas import NewsItem, RecommendNewsResponse

logger = logging.getLogger(__name__)


def recommend_news(
    recent_queries: List[str],
    preferred_tags: Optional[List[str]],
    top_k: int,
    collection: str,
) -> RecommendNewsResponse:
    embedding_client = get_embedding_client()
    milvus_client = get_milvus_client()

    query_text = "。".join(recent_queries)
    query_vec = embedding_client.embed_texts([query_text])
    results = milvus_client.search(collection, query_vec, top_k=top_k)[0]

    items: List[NewsItem] = []
    for hit in results:
        meta = hit["metadata"]
        tags = meta.get("tags", [])
        if preferred_tags:
            if not any(t in tags for t in preferred_tags):
                continue

        items.append(
            NewsItem(
                id=str(meta.get("id", "")),
                title=meta.get("title", "Untitled"),
                content_snippet=(meta.get("content", "") or "")[:200],
                tags=tags,
                published_at=str(meta.get("published_at", "")),
                score=float(hit["score"]),
            )
        )

    return RecommendNewsResponse(items=items)
