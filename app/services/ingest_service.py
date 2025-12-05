from typing import List, Tuple
import logging
from pathlib import Path

from app.embeddings.embedding_client import get_embedding_client
from app.vectorstore.milvus_client import get_milvus_client
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

    success_count = 0
    failed_files: List[str] = []

    milvus_client.create_collection_if_not_exists(collection_name)

    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            failed_files.append(path_str)
            continue

        try:
            raw_text = parse_file(path)
            text = clean_text(raw_text)
            chunks = simple_chunk(text, max_tokens=chunk_size, overlap=overlap)
            vectors = embedding_client.embed_texts(chunks)
            metadatas = [
                {"doc_id": str(path), "chunk_id": i, "text": c}
                for i, c in enumerate(chunks)
            ]
            milvus_client.insert_vectors(collection_name, vectors, metadatas)
            success_count += 1
        except Exception as e:
            logger.exception(f"Failed to ingest file={path}: {e}")
            failed_files.append(path_str)

    return success_count, failed_files
