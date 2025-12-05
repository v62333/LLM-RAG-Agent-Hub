from typing import List
import logging

from app.embeddings.embedding_client import get_embedding_client
from app.vectorstore.milvus_client import get_milvus_client
from app.llm.llm_client import get_llm_client
from app.models.schemas import SourceChunk, RagAnswer

logger = logging.getLogger(__name__)


def retrieve_top_k(
    question: str,
    collection: str,
    top_k: int = 5,
) -> List[SourceChunk]:
    embedding_client = get_embedding_client()
    milvus_client = get_milvus_client()

    query_vec = embedding_client.embed_texts([question])
    results = milvus_client.search(collection, query_vec, top_k=top_k)[0]

    chunks: List[SourceChunk] = []
    for i, hit in enumerate(results):
        meta = hit["metadata"]
        chunks.append(
            SourceChunk(
                doc_id=str(meta.get("doc_id", "")),
                doc_name=meta.get("doc_id"),
                chunk_id=int(meta.get("chunk_id", i)),
                score=float(hit["score"]),
                snippet=(meta.get("text", "") or "")[:200],
            )
        )
    return chunks


def build_context(chunks: List[SourceChunk]) -> str:
    texts = [f"[{c.doc_id}#{c.chunk_id}] {c.snippet}" for c in chunks]
    return "\n\n".join(texts)


def build_prompt_with_context(question: str, context: str) -> str:
    return (
        "以下是與問題相關的文件內容：\n"
        f"{context}\n\n"
        "請根據上述內容，嚴謹回答使用者問題，若資料不足請明講。\n"
        f"問題：{question}"
    )


async def answer_with_rag(
    question: str,
    collection: str,
    top_k: int = 5,
) -> RagAnswer:
    client = get_llm_client()
    chunks = retrieve_top_k(question, collection, top_k=top_k)
    context = build_context(chunks)
    prompt = build_prompt_with_context(question, context)

    llm_result = await client.generate(
        system_prompt="你是一位嚴謹的金融 / 技術文件說明助手。",
        user_prompt=prompt,
        temperature=0.2,
        max_tokens=512,
    )

    return RagAnswer(
        answer=llm_result["output"],
        strategy="rag",
        sources=chunks,
        metadata={"model": llm_result["model"]},
    )
