from typing import List, Dict, Any, Optional
import logging

from app.embeddings.embedding_client import get_embedding_client
from app.vectorstore.milvus_client import get_milvus_client
from app.llm.llm_client import get_llm_client
from app.utils.es_client import search_bm25, get_es_client
from app.models.schemas import SourceChunk, RagAnswer

logger = logging.getLogger(__name__)


def retrieve_top_k(
    question: str,
    collection: str,
    top_k: int = 5,
) -> List[SourceChunk]:
    """
    純向量檢索 (Vector Search Only)
    """
    embedding_client = get_embedding_client()
    milvus_client = get_milvus_client()

    # 1. 轉向量
    query_vec = embedding_client.embed_texts([question])
    
    # 2. 搜尋 Milvus
    # 注意: search 回傳的是 List[List[hit]]，我們只查一句所以取 [0]
    results = milvus_client.search(collection, query_vec, top_k=top_k)[0]

    chunks: List[SourceChunk] = []
    for i, hit in enumerate(results):
        meta = hit["metadata"]
        # 容錯處理：確保 metadata 欄位存在
        doc_id = str(meta.get("doc_id", ""))
        chunk_id = int(meta.get("chunk_id", i))
        text = meta.get("text", "")
        
        chunks.append(
            SourceChunk(
                doc_id=doc_id,
                doc_name=doc_id, # 暫時用 doc_id 當 name
                chunk_id=chunk_id,
                score=float(hit["score"]),
                snippet=text[:500] if text else "", # 限制 snippet 長度
            )
        )
    return chunks


def retrieve_hybrid(
    question: str, 
    collection: str, 
    top_k: int = 5, 
    rrf_k: int = 60
) -> List[SourceChunk]:
    """
    混合檢索 (Hybrid Search)
    流程: 向量檢索 + BM25 檢索 -> RRF 融合排序 -> 回補缺失文本 -> 回傳 SourceChunk
    """
    # 1. 擴大檢索範圍 (Fetch more candidates)
    # 為了讓重排序有效，通常會抓取 top_k 的 2~3 倍資料進來做融合
    candidate_limit = top_k * 2
    
    # 2. 執行雙路檢索
    # Route A: Vector Search (會回傳完整的 SourceChunk，包含 text)
    vec_chunks = retrieve_top_k(question, collection, top_k=candidate_limit)
    
    # Route B: BM25 Search (通常只回傳 doc_id, chunk_id, score)
    bm25_hits = search_bm25(collection, question, size=candidate_limit)

    # 3. 建立 RRF 分數表 (Reciprocal Rank Fusion)
    rrf_scores: Dict[str, float] = {}
    
    # 用來暫存已經抓到的完整 Chunk 物件 (來自 Vector)，避免重複查詢資料庫
    chunk_cache: Dict[str, SourceChunk] = {}

    # 3.1 計算 Vector RRF
    for rank, chunk in enumerate(vec_chunks):
        unique_id = f"{chunk.doc_id}#{chunk.chunk_id}"
        rrf_scores[unique_id] = rrf_scores.get(unique_id, 0) + 1 / (rrf_k + rank + 1)
        chunk_cache[unique_id] = chunk

    # 3.2 計算 BM25 RRF
    for rank, (doc_id, chunk_id, _) in enumerate(bm25_hits):
        unique_id = f"{doc_id}#{chunk_id}"
        rrf_scores[unique_id] = rrf_scores.get(unique_id, 0) + 1 / (rrf_k + rank + 1)
        # 注意：BM25 hits 沒有 text，先不放入 chunk_cache

    # 4. 排序並取最終 Top K
    # 依照 RRF 分數由高到低排序
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # 5. 結果組裝與缺失資料回補 (Data Hydration)
    final_chunks: List[SourceChunk] = []
    missing_ids: List[str] = [] # 那些只出現在 BM25 的幸運兒

    # 先分類：哪些 cache 有，哪些 cache 沒有
    for unique_id, score in sorted_candidates:
        if unique_id in chunk_cache:
            # Vector 已經抓過這個 chunk，直接用，並更新 score 為 RRF 分數
            chunk = chunk_cache[unique_id]
            chunk.score = score 
            final_chunks.append(chunk)
        else:
            # 這是 BM25 獨有的 chunk，我們只有 ID，需要去撈文本
            missing_ids.append(unique_id)

    # 6. 批量補抓缺失的文本 (mget 優化版)
    if missing_ids:
        logger.info(f"Hybrid Search: Fetching {len(missing_ids)} missing chunks from ES.")
        es = get_es_client()
        index_name = f"{es.cfg.ES_INDEX_PREFIX}_{collection}"
        
        try:
            # 使用 mget 一次取回所有缺失文件，避免迴圈連線
            resp = es.client.mget(index=index_name, body={"ids": missing_ids})
            
            for doc in resp.get('docs', []):
                if doc.get('found'):
                    source = doc['_source']
                    mid = doc['_id'] # 這裡的 _id 就是我們傳進去的 doc_id#chunk_id
                    
                    final_chunks.append(SourceChunk(
                        doc_id=source["doc_id"],
                        doc_name=source["doc_id"],
                        chunk_id=source["chunk_id"],
                        score=rrf_scores.get(mid, 0.0),
                        snippet=source["text"][:500] # 取 snippet
                    ))
                else:
                    logger.warning(f"Chunk missing in ES: {doc.get('_id')}")
                    
        except Exception as e:
            logger.error(f"Failed to hydrate chunks via mget: {e}")

    # 再次依照 RRF 分數排序確保順序正確 (因為補抓的資料是後來 append 的)
    final_chunks.sort(key=lambda x: x.score, reverse=True)
    
    return final_chunks


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
    use_hybrid: bool = False, # 接收來自 Router 的開關
) -> RagAnswer:
    
    # 1. 根據開關決定檢索策略
    if use_hybrid:
        chunks = retrieve_hybrid(question, collection, top_k=top_k)
        strategy = "hybrid"
    else:
        chunks = retrieve_top_k(question, collection, top_k=top_k)
        strategy = "vector"

    # 2. 構建 Context 與 Prompt
    context = build_context(chunks)
    prompt = build_prompt_with_context(question, context)

    # 3. 呼叫 LLM
    client = get_llm_client()
    llm_result = await client.generate(
        system_prompt="你是一位嚴謹的金融 / 技術文件說明助手。",
        user_prompt=prompt,
        temperature=0.2,
        max_tokens=1024,
    )

    return RagAnswer(
        answer=llm_result["output"],
        strategy=strategy,
        sources=chunks,
        metadata={"model": llm_result["model"]},
    )