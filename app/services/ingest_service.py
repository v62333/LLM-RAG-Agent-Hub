from typing import List, Tuple
import logging
from pathlib import Path

from app.embeddings.embedding_client import get_embedding_client
from app.vectorstore.milvus_client import get_milvus_client
# [修改點 1] 引入我們剛寫好的 ES Client
from app.utils.es_client import get_es_client

from app.utils.text_cleaning import clean_text
from app.utils.chunking import simple_chunk

logger = logging.getLogger(__name__)


def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    logger.info(f"Fallback text read for file: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def ingest_files_to_collection(
    file_paths: List[str],
    collection_name: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> Tuple[int, List[str]]:
    embedding_client = get_embedding_client()
    milvus_client = get_milvus_client()
    
    # [修改點 2] 初始化 ES Client
    es_client = get_es_client()

    success_count = 0
    failed_files: List[str] = []

    milvus_client.create_collection_if_not_exists(collection_name)
    # 註: ES 的 Index 會在第一次 index_doc 時自動建立 (Lazy Create)，所以這裡不用特別呼叫

    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            failed_files.append(path_str)
            continue

        try:
            raw_text = parse_file(path)
            text = clean_text(raw_text)
            chunks = simple_chunk(text, max_tokens=chunk_size, overlap=overlap)
            
            # --- Step A: 寫入 Milvus (Vector) ---
            vectors = embedding_client.embed_texts(chunks)
            metadatas = [
                {"doc_id": str(path), "chunk_id": i, "text": c}
                for i, c in enumerate(chunks)
            ]
            milvus_client.insert_vectors(collection_name, vectors, metadatas)

            # --- [修改點 3] Step B: 同步寫入 Elasticsearch (Keyword/BM25) ---
            # 遍歷所有的 chunks，將它們也存入 ES
            # 這樣保證了同一份 chunk 既有 Vector 也有 BM25 索引
            for i, chunk_text in enumerate(chunks):
                es_client.index_doc(
                    collection=collection_name,
                    doc_id=str(path),  # 保持與 Milvus 一致的 doc_id
                    chunk_id=i,        # 保持與 Milvus 一致的 chunk_id
                    text=chunk_text
                )

            success_count += 1
            logger.info(f"Successfully ingested file: {path} (Milvus + ES)")

        except Exception as e:
            # 如果 Milvus 成功但 ES 失敗 (或反之)，我們會捕捉到 Exception
            # 並將此檔案列為失敗，這在資料一致性上是比較安全的做法
            logger.exception(f"Failed to ingest file={path}: {e}")
            failed_files.append(path_str)

    return success_count, failed_files