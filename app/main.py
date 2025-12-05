from fastapi import FastAPI
from app.api import (
    routes_health,
    routes_prompt,
    routes_embed,
    routes_ingest,
    routes_rag,
    routes_agent,
    routes_recommend,
)
from app.core.config import settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    """建立並回傳 FastAPI 主應用。"""
    setup_logging()
    app = FastAPI(
        title="LLM-RAG Agent Hub",
        description="金融知識 & 廣告優化多 Agent 平台",
        version="0.1.0",
    )

    app.include_router(routes_health.router, prefix="/health", tags=["health"])
    app.include_router(routes_prompt.router, prefix="/prompt", tags=["prompt"])
    app.include_router(routes_embed.router, prefix="/embed", tags=["embed"])
    app.include_router(routes_ingest.router, prefix="/ingest", tags=["ingest"])
    app.include_router(routes_rag.router, prefix="/rag", tags=["rag"])
    app.include_router(routes_agent.router, prefix="/agent", tags=["agent"])
    app.include_router(routes_recommend.router, prefix="/recommend", tags=["recommend"])

    return app


app = create_app()
