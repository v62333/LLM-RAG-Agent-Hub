from typing import List, Optional
import numpy as np
import logging

from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """封裝 Embedding 模型（例如 BGE-base-zh）。"""

    def __init__(self) -> None:
        self.model_name = settings.embedding_model_name
        self._model: Optional[SentenceTransformer] = None

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        self._ensure_model_loaded()
        if not texts:
            return np.zeros((0, 768), dtype="float32")
        embeddings = self._model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")


_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
