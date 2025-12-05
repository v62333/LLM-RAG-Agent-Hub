from fastapi import APIRouter
from app.models.schemas import RecommendNewsRequest, RecommendNewsResponse
from app.services.recommend_service import recommend_news
from app.core.config import settings

router = APIRouter()


@router.post("/news", response_model=RecommendNewsResponse)
async def recommend_news_api(req: RecommendNewsRequest) -> RecommendNewsResponse:
    return recommend_news(
        recent_queries=req.recent_queries,
        preferred_tags=req.preferred_tags or [],
        top_k=req.top_k,
        collection=settings.news_collection,
    )
