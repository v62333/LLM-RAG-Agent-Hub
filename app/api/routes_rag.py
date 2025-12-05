from fastapi import APIRouter
from app.models.schemas import RagAskRequest, RagAskResponse
from app.services.rag_service import answer_with_rag
from app.services.graph_rag_service import answer_with_graph_rag

router = APIRouter()


@router.post("/ask", response_model=RagAskResponse)
async def rag_ask(req: RagAskRequest) -> RagAskResponse:
    answer = await answer_with_rag(
        question=req.question,
        collection=req.collection.value,
        top_k=req.top_k,
    )
    return RagAskResponse(result=answer)


@router.post("/graph_ask", response_model=RagAskResponse)
async def rag_graph_ask(req: RagAskRequest) -> RagAskResponse:
    answer = await answer_with_graph_rag(
        question=req.question,
        collection=req.collection.value,
        top_k=req.top_k,
    )
    return RagAskResponse(result=answer)
