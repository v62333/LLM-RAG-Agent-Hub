from typing import List, Dict, Any, Optional
import logging

import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class MilvusClient:
    """負責與 Milvus 溝通。"""

    def __init__(self) -> None:
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.connected = False

    def connect(self) -> None:
        if self.connected:
            return
        logger.info(f"Connecting to Milvus at {self.host}:{self.port}")
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            user=settings.milvus_user or None,
            password=settings.milvus_password or None,
            secure=settings.milvus_secure,
        )
        self.connected = True

    def _get_collection(self, name: str) -> Collection:
        self.connect()
        return Collection(name)

    def create_collection_if_not_exists(self, name: str, dim: int = 768) -> None:
        self.connect()
        if utility.has_collection(name):
            return

        logger.info(f"Creating Milvus collection: {name}")
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="doc_id",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=2048,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
            ),
        ]
        schema = CollectionSchema(fields=fields, description=f"Collection for {name}")
        collection = Collection(name=name, schema=schema)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

    def insert_vectors(
        self,
        collection: str,
        vectors: np.ndarray,
        metadatas: List[Dict[str, Any]],
    ) -> List[str]:
        self.create_collection_if_not_exists(collection, dim=vectors.shape[1])
        col = self._get_collection(collection)

        if len(metadatas) != len(vectors):
            raise ValueError("metadatas and vectors length mismatch")

        doc_ids = [str(m.get("doc_id", "")) for m in metadatas]
        chunk_ids = [int(m.get("chunk_id", i)) for i, m in enumerate(metadatas)]
        texts = [m.get("text", "") for m in metadatas]
        embeddings = vectors.tolist()

        logger.info(f"Inserting {len(metadatas)} vectors into {collection}")
        res = col.insert(
            [
                doc_ids,
                chunk_ids,
                texts,
                embeddings,
            ],
            fields=["doc_id", "chunk_id", "text", "embedding"],
        )
        col.flush()
        ids = [str(x) for x in res.primary_keys]
        return ids

    def search(
        self,
        collection: str,
        query_vectors: np.ndarray,
        top_k: int = 5,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        col = self._get_collection(collection)
        col.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
        logger.info(f"Searching Milvus collection={collection}, top_k={top_k}")
        results = col.search(
            data=query_vectors.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id", "chunk_id", "text"],
        )

        wrapped: List[List[Dict[str, Any]]] = []
        for hits in results:
            cur_list: List[Dict[str, Any]] = []
            for hit in hits:
                meta = {
                    "doc_id": hit.entity.get("doc_id"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "text": hit.entity.get("text"),
                }
                cur_list.append(
                    {
                        "id": str(hit.id),
                        "score": float(hit.distance),
                        "metadata": meta,
                    }
                )
            wrapped.append(cur_list)
        return wrapped


_milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient()
    return _milvus_client
