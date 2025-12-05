from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM backend: "openai" 或 "local"
    llm_backend: str = "local"

    # OpenAI 相容設定（若使用雲端）
    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model_name: str = "gpt-4o-mini"

    # 本地 LLM（例如 Ollama + Qwen2.5）
    local_llm_base_url: str = "http://localhost:11434"
    local_llm_model_name: str = "qwen2.5:7b"

    # Embedding 模型
    embedding_model_name: str = "BAAI/bge-base-zh-v1.5"

    # Milvus 設定
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_secure: bool = False

    # Collection 名稱
    docs_collection: str = "finance_docs"
    news_collection: str = "finance_news"
    custom_collection: str = "custom_collection"

    # 路徑
    data_dir: str = "data"
    docs_dir: str = "data/docs"
    news_dir: str = "data/news"
    ads_dir: str = "data/ads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
