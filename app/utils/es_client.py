import logging
from typing import List, Tuple, Dict, Any, Optional
from elasticsearch import Elasticsearch, NotFoundError, ConnectionError
from app.core.config import get_settings

# 設定 Log
logger = logging.getLogger(__name__)

class ESClientWrapper:
    """
    Elasticsearch 客戶端封裝類別
    負責連線管理、索引初始化、文件寫入與檢索
    """
    _instance = None

    def __init__(self):
        self.cfg = get_settings()
        self.client = Elasticsearch(
            hosts=[self.cfg.ES_HOST],
            # 正式環境建議設定：
            max_retries=3,
            retry_on_timeout=True,
            request_timeout=30
        )
        self._check_connection()

    @classmethod
    def get_instance(cls):
        """實作 Singleton 模式，確保全域只有一個連線實例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _check_connection(self):
        """測試連線是否成功"""
        try:
            if not self.client.ping():
                logger.error("Could not connect to Elasticsearch!")
            else:
                logger.info(f"Successfully connected to Elasticsearch at {self.cfg.ES_HOST}")
        except Exception as e:
            logger.error(f"Elasticsearch connection error: {e}")

    def _get_index_name(self, collection: str) -> str:
        return f"{self.cfg.ES_INDEX_PREFIX}_{collection}"

    def ensure_index_exists(self, collection: str):
        """
        關鍵：確保索引存在，並設定正確的 Mapping。
        若沒有設定 Mapping，ES 會自動猜測型別，導致 BM25 效果不穩。
        """
        index_name = self._get_index_name(collection)
        
        if self.client.indices.exists(index=index_name):
            return

        # 定義 Schema
        # 注意：如果你是中文內容，建議將 "analyzer": "standard" 改為 "analyzer": "ik_max_word"
        # 前提是你的 ES 必須安裝 IK Analysis Plugin
        body = {
            "settings": {
                "number_of_shards": 1,  # 小數據量建議設為 1，效能較好
                "number_of_replicas": 0 # 單機部署設為 0，Cluster 設為 1
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text", 
                        "analyzer": "standard" # <--- 改成 "ik_max_word" 如果你有裝插件
                    },
                    "doc_id": {
                        "type": "keyword" # keyword 不分詞，用於精確過濾
                    },
                    "chunk_id": {
                        "type": "integer"
                    },
                    "created_at": {
                        "type": "date"
                    }
                }
            }
        }

        try:
            self.client.indices.create(index=index_name, body=body)
            logger.info(f"Created index: {index_name} with mapping.")
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")

    def index_doc(self, collection: str, doc_id: str, chunk_id: int, text: str):
        """
        寫入單一文件
        """
        # 1. 確保索引存在 (Lazy Initialization)
        self.ensure_index_exists(collection)
        
        index_name = self._get_index_name(collection)
        unique_id = f"{doc_id}#{chunk_id}"

        try:
            self.client.index(
                index=index_name,
                id=unique_id,
                body={
                    "text": text,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id
                }
            )
            # logger.debug(f"Indexed document {unique_id}") 
        except Exception as e:
            logger.error(f"Error indexing document {unique_id}: {e}")
            raise e

    def search_bm25(self, collection: str, query: str, size: int = 20) -> List[Tuple[str, int, float]]:
        """
        執行 BM25 檢索
        Returns: List of (doc_id, chunk_id, score)
        """
        index_name = self._get_index_name(collection)

        # 定義查詢語法
        search_body = {
            "query": {
                "match": {
                    "text": {
                        "query": query,
                        "operator": "OR"  # 或是 "AND"，視需求而定
                    }
                }
            },
            "_source": ["doc_id", "chunk_id"], # 只取需要的欄位，節省頻寬
            "size": size
        }

        try:
            response = self.client.search(index=index_name, body=search_body)
            
            hits = response.get("hits", {}).get("hits", [])
            return [
                (
                    hit["_source"]["doc_id"],
                    hit["_source"]["chunk_id"],
                    hit["_score"] # 回傳 BM25 分數
                )
                for hit in hits
            ]

        except NotFoundError:
            logger.warning(f"Index {index_name} does not exist. Returning empty results.")
            return []
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

# --- 對外暴露的 Helper Functions (保持你原本的調用介面) ---

def get_es_client():
    return ESClientWrapper.get_instance()

def index_doc(collection: str, doc_id: str, chunk_id: int, text: str):
    get_es_client().index_doc(collection, doc_id, chunk_id, text)

def search_bm25(collection: str, query: str, size: int = 20) -> List[Tuple[str, int, float]]:
    return get_es_client().search_bm25(collection, query, size)