from typing import Optional, Dict, Any
from app.models.enums import Domain
from app.llm.llm_client import get_llm_client


def get_system_prompt_for_domain(domain: Domain) -> str:
    if domain == Domain.finance:
        return "你是一位嚴謹的金融知識顧問，回答時需根據資料與風險揭露。"
    if domain == Domain.ads:
        return "你是一位數據驅動的廣告優化專家，擅長分析指標與提出具體建議。"
    return "你是一位 helpful 且穩定的 AI 助理。"


async def generate_with_domain(
    user_prompt: str,
    domain: Domain,
    system_prompt_override: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = get_llm_client()
    system_prompt = system_prompt_override or get_system_prompt_for_domain(domain)
    return await client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_params=extra_params,
    )
