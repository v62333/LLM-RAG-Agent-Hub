from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient(ABC):    #指定所有LLM的規格書
    @abstractmethod
    async def generate(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """使用 OpenAI 相容 API。"""

    def __init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai 套件未安裝，但 llm_backend 設為 'openai'。請先 `pip install openai`。"
            ) from e
        if not settings.llm_api_key:
            logger.warning("LLM_API_KEY 未設置，請在 .env 中設定。")
        self._OpenAI = OpenAI
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base,
        )
        self.model = settings.llm_model_name

    async def generate(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_params:
            params.update(extra_params)

        resp = self.client.chat.completions.create(**params)
        choice = resp.choices[0]
        output = choice.message.content or ""
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return {
            "output": output,
            "model": self.model,
            "usage": usage,
        }


class LocalLLMClient(LLMClient):
    """呼叫本地 LLM（例如 Ollama + Qwen2.5）。"""

    def __init__(self) -> None:
        self.base_url = settings.local_llm_base_url.rstrip("/")
        self.model = settings.local_llm_model_name

    async def generate(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = user_prompt
        if system_prompt:
            prompt = f"[指示]\n{system_prompt}\n\n[使用者提問]\n{user_prompt}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        logger.info(f"Calling Local LLM model={self.model} at {self.base_url}")
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        output = data.get("response", "")

        return {
            "output": output,
            "model": self.model,
            "usage": None,
        }


_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        if settings.llm_backend == "local":
            logger.info("Using LocalLLMClient (e.g., Qwen2.5)")
            _llm_client = LocalLLMClient()
        else:
            logger.info("Using OpenAIClient")
            _llm_client = OpenAIClient()
    return _llm_client
