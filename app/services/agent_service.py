from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import pandas as pd

from app.storage.file_storage import get_storage
from app.llm.llm_client import get_llm_client
from app.models.schemas import AgentRunResponse, AgentStepResult

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: str

    @abstractmethod
    async def run(self, **kwargs) -> Dict[str, Any]:
        ...


class DataAgent(BaseAgent):
    name = "DataAgent"

    async def run(self, **kwargs) -> Dict[str, Any]:
        storage = get_storage()
        csv_path = storage.get_ads_csv_path()
        logger.info(f"Loading ads data from {csv_path}")
        df = pd.read_csv(csv_path)

        date_start: Optional[str] = kwargs.get("date_start")
        date_end: Optional[str] = kwargs.get("date_end")

        if date_start:
            df = df[df["date"] >= date_start]
        if date_end:
            df = df[df["date"] <= date_end]

        df["ctr"] = df["clicks"] / df["impressions"].clip(lower=1)
        df["cpc"] = df["spend"] / df["clicks"].clip(lower=1)
        df["cpa"] = df["spend"] / df["conversions"].clip(lower=1)

        agg = (
            df.groupby("campaign_name")
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                conversions=("conversions", "sum"),
                spend=("spend", "sum"),
                ctr=("ctr", "mean"),
                cpc=("cpc", "mean"),
                cpa=("cpa", "mean"),
            )
            .reset_index()
        )

        summary = agg.to_markdown(index=False)

        return {
            "summary": summary,
            "raw_df_head": df.head().to_dict(),
        }


class AnalysisAgent(BaseAgent):
    name = "AnalysisAgent"

    async def run(self, **kwargs) -> Dict[str, Any]:
        client = get_llm_client()
        data_summary = kwargs.get("data_summary", "")
        prompt = f"以下是廣告數據摘要：\n{data_summary}\n\n請指出表現最好與最差的 campaign 並簡要說明原因。"
        result = await client.generate(
            system_prompt="你是一位數據分析顧問。",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=512,
        )
        return {"analysis": result["output"]}


class AdOptimizationAgent(BaseAgent):
    name = "AdOptimizationAgent"

    async def run(self, **kwargs) -> Dict[str, Any]:
        client = get_llm_client()
        analysis_text = kwargs.get("analysis", "")
        prompt = (
            "以下是廣告表現分析：\n"
            f"{analysis_text}\n\n"
            "請根據上述內容，給出 3-5 點具體的廣告優化建議，"
            "包含：應調整的 campaign、預算調整方向、素材或受眾建議。"
        )
        result = await client.generate(
            system_prompt="你是一位資深成效型廣告優化專家。",
            user_prompt=prompt,
            temperature=0.4,
            max_tokens=512,
        )
        return {"suggestions": result["output"]}


class AgentOrchestrator:
    def __init__(self) -> None:
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.opt_agent = AdOptimizationAgent()

    async def run_flow(
        self,
        task: str,
        date_start: Optional[str],
        date_end: Optional[str],
    ) -> AgentRunResponse:
        steps: List[AgentStepResult] = []

        data_result = await self.data_agent.run(date_start=date_start, date_end=date_end)
        data_summary = data_result.get("summary", "")
        steps.append(
            AgentStepResult(
                name=self.data_agent.name,
                summary=data_summary,
                raw_output=data_result,
            )
        )

        analysis_result = await self.analysis_agent.run(data_summary=data_summary)
        analysis_text = analysis_result.get("analysis", "")
        steps.append(
            AgentStepResult(
                name=self.analysis_agent.name,
                summary=analysis_text[:200],
                raw_output=analysis_result,
            )
        )

        opt_result = await self.opt_agent.run(analysis=analysis_text)
        suggestions_text = opt_result.get("suggestions", "")
        steps.append(
            AgentStepResult(
                name=self.opt_agent.name,
                summary=suggestions_text[:200],
                raw_output=opt_result,
            )
        )

        return AgentRunResponse(
            data_summary=data_summary,
            analysis_insights=analysis_text,
            optimization_suggestions=suggestions_text,
            steps=steps,
        )


_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
