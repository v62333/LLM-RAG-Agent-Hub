from fastapi import APIRouter
from app.models.schemas import PromptRequest, PromptResponse
from app.services.prompt_service import generate_with_domain

router = APIRouter()


@router.post("/", response_model=PromptResponse)
async def prompt_api(req: PromptRequest) -> PromptResponse:
    result = await generate_with_domain(
        user_prompt=req.user_prompt,
        domain=req.domain,
        system_prompt_override=req.system_prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return PromptResponse(
        output=result["output"],
        model=result["model"],
        usage=result.get("usage"),
    )
