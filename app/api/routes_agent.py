from fastapi import APIRouter
from app.models.schemas import AgentTaskRequest, AgentRunResponse
from app.services.agent_service import get_orchestrator

router = APIRouter()


@router.post("/run", response_model=AgentRunResponse)
async def agent_run(req: AgentTaskRequest) -> AgentRunResponse:
    orchestrator = get_orchestrator()
    return await orchestrator.run_flow(
        task=req.task,
        date_start=req.date_start,
        date_end=req.date_end,
    )
