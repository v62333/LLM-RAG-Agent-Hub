from fastapi import APIRouter
from app.models.schemas import RagAskRequest, RagAskResponse
from app.services.rag_service import answer_with_rag
from app.services.graph_rag_service import answer_with_graph_rag

router = APIRouter()


@router.post("/ask", response_model=RagAskResponse)
async def rag_ask(req: RagAskRequest) -> RagAskResponse:
    # 修改：明確傳入 use_hybrid 參數，實作快速切換
    answer = await answer_with_rag(
        question=req.question,
        collection=req.collection.value,
        top_k=req.top_k,
        use_hybrid=req.use_hybrid,  # <--- 關鍵修改：將前端開關傳入 Service
    )
    return RagAskResponse(result=answer)


@router.post("/graph_ask", response_model=RagAskResponse)
async def rag_graph_ask(req: RagAskRequest) -> RagAskResponse:
    # GraphRAG 暫時維持原樣，除非 GraphRAG 也要支援混合檢索
    answer = await answer_with_graph_rag(
        question=req.question,
        collection=req.collection.value,
        top_k=req.top_k,
    )
    return RagAskResponse(result=answer)