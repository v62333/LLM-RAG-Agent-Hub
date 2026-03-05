from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
import pandas as pd
import re
import json

from pydantic import ValidationError

from app.models.schemas import (
    AgentRunResponse,
    AgentStepResult,
    OptimizationItem,
    AgentTaskRequest,
    QualityEvaluation
)
from app.storage.file_storage import get_storage
from app.llm.llm_client import get_llm_client

logger = logging.getLogger(__name__)


# =========================
# BaseAgent
# =========================

class BaseAgent(ABC):
    name: str

    @abstractmethod
    async def run(self, **kwargs) -> Dict[str, Any]:
        ...


# =========================
# DataAgent
# =========================

class DataAgent(BaseAgent):
    name = "DataAgent"

    async def run(self, **kwargs) -> Dict[str, Any]:
        try:
            req = AgentTaskRequest(**kwargs)
        except ValidationError as e:
            error_msg = e.errors()[0]["msg"]
            return {"summary": "數據讀取任務中止", "error": error_msg, "verified": False}

        try:
            storage = get_storage()
            csv_path = storage.get_ads_csv_path()
            df = pd.read_csv(csv_path)

            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            if req.date_start:
                df = df[df["date"] >= pd.to_datetime(req.date_start)]
            if req.date_end:
                df = df[df["date"] <= pd.to_datetime(req.date_end)]

            agg = (
                df.groupby("campaign_name")
                .agg(
                    impressions=("impressions", "sum"),
                    clicks=("clicks", "sum"),
                    conversions=("conversions", "sum"),
                    spend=("spend", "sum"),
                )
                .reset_index()
            )

            agg["ctr"] = agg["clicks"] / agg["impressions"].clip(lower=1)
            agg["cpc"] = agg["spend"] / agg["clicks"].clip(lower=1)
            agg["cpa"] = agg["spend"] / agg["conversions"].clip(lower=1)

            summary = agg.to_markdown(index=False)

            return {
                "summary": summary,
                "raw_df_head": df.head().to_dict(),
                "verified": True
            }

        except Exception as e:
            logger.error(f"[{self.name}] error: {e}", exc_info=True)
            return {"summary": "系統執行錯誤", "error": str(e), "verified": False}


# =========================
# AnalysisAgent
# =========================

class AnalysisAgent(BaseAgent):
    name = "AnalysisAgent"

    async def run(self, **kwargs) -> Dict[str, Any]:
        client = get_llm_client()
        data_summary = kwargs.get("data_summary", "")

        if not data_summary:
            return {"analysis": "無數據可分析"}

        prompt = (
            f"以下是廣告數據摘要：\n{data_summary}\n\n"
            "請指出表現最好與最差的 campaign 並簡要說明原因。"
        )

        result = await client.generate(
            system_prompt="你是一位數據分析顧問。",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=512,
        )

        return {"analysis": result.get("output", "")}


# =========================
# OptimizationAgent
# =========================

class OptimizationAgent(BaseAgent):
    name = "OptimizationAgent"

    async def run(self, **kwargs) -> Dict[str, Any]:
        client = get_llm_client()

        analysis_text = kwargs.get("analysis", "")
        # 🟢 融合版保留：正確接住來自 Orchestrator 的 data_summary
        data_summary = kwargs.get("data_summary", "")  

        sys_prompt = (
            "你是一位資深成效型廣告優化專家。\n"
            "建議必須包含：目標對象、行動方案、預期成效。\n"
            "嚴禁模糊建議。"
        )

        max_retries = 3
        final_markdown = ""
        last_error_hint = ""

        for attempt in range(max_retries):
            try:
                user_prompt = (
                    "請提供 3~5 點優化建議。\n"
                    "使用 Markdown 編號清單。\n"
                    "每點需包含：\n"
                    "- 目標對象:\n"
                    "- 行動方案:\n"
                    "- 預期成效:\n\n"
                    f"{analysis_text}"
                )

                result = await client.generate(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3 + attempt * 0.1,
                    max_tokens=1024,
                )

                final_markdown = result.get("output", "")

                success, parsed_or_error = self._parse_and_validate(final_markdown)

                if not success:
                    last_error_hint = parsed_or_error
                    continue

                # 🟢 融合版保留：正確傳入 data_summary 給品質評估員
                eval_result = await self._evaluate_quality(
                    client,
                    data_summary,
                    analysis_text,
                    final_markdown
                )

                if not eval_result.passed:
                    last_error_hint = eval_result.critique
                    continue

                return {
                    "suggestions": final_markdown,
                    "structured_data": [
                        item.model_dump() for item in parsed_or_error
                    ],
                    "evaluation": eval_result.model_dump(),
                    "verified": True
                }

            except Exception as e:
                last_error_hint = str(e)

        return {
            "suggestions": final_markdown,
            "structured_data": [],
            "verified": False,
            "fail_reason": last_error_hint
        }

    async def _evaluate_quality(
        self,
        client,
        data_summary: str,
        analysis_source: str,
        generated_content: str
    ) -> QualityEvaluation:

        eval_prompt = (
            "### 任務描述\n"
            "你是一位嚴格的廣告成效稽核員。\n"
            "請同時對照「原始數據摘要」與「分析內容」評核「生成的建議」。\n\n"

            "### 評分標準 (總分 100):\n"
            "1. **事實忠實度 (50%)**:\n"
            "   - 建議中的數據或結論必須能在原始數據摘要或分析內容中找到依據。\n"
            "   - 若出現原始資料未包含的數字或 campaign，視為幻覺。\n"
            "2. **執行具體性 (30%)**:\n"
            "   - 必須包含「目標對象、行動方案、預期成效」。\n"
            "3. **邏輯連貫性 (20%)**:\n"
            "   - 建議是否能回應分析中指出的問題？\n\n"

            "### 輸入內容\n"
            f"[原始數據摘要]:\n{data_summary}\n\n"
            f"[分析內容]:\n{analysis_source}\n\n"
            f"[待評核建議]:\n{generated_content}\n\n"

            "### 輸出要求\n"
            "僅回傳 JSON：\n"
            "{\n"
            "  \"score\": <int>,\n"
            "  \"breakdown\": {\"faithfulness\": int, \"concreteness\": int, \"logic\": int},\n"
            "  \"critique\": \"<簡短扣分原因>\",\n"
            "  \"hallucination_detected\": <bool>\n"
            "}"
        )

        try:
            res = await client.generate(
                system_prompt="你是嚴格審核員。",
                user_prompt=eval_prompt,
                temperature=0.1,
            )

            raw = res.get("output", "").strip()
            json_match = re.search(r"\{.*?\}", raw, re.DOTALL)

            if not json_match:
                raise ValueError("無法解析 JSON")

            data = json.loads(json_match.group(0))
            score = int(data.get("score", 0))

            return QualityEvaluation(
                score=score,
                critique=data.get("critique", ""),
                passed=score >= 80
            )

        except Exception:
            return QualityEvaluation(
                score=0,
                critique="評分解析失敗",
                passed=False
            )

    def _parse_and_validate(self, text: str) -> Tuple[bool, Any]:
        blocks = re.split(r"(?:^|\n)\d+\.\s+", text)
        blocks = [b.strip() for b in blocks if b.strip()]

        if not (3 <= len(blocks) <= 5):
            return False, "建議數量需為 3~5 點"

        items = []

        for i, block in enumerate(blocks):
            clean = block.replace("\n", " ")

            # 🟢 融合版保留：版本 1 的強健 Regex，避免 LLM 的 Markdown 干擾解析
            target = re.search(r"(?:\[|\*\*|^)?目標對象(?:\]|\*\*)?[:：]\s*(.*?)(?=\s*(?:\[|\*\*|^)?行動方案|$)", clean)
            action = re.search(r"(?:\[|\*\*|^)?行動方案(?:\]|\*\*)?[:：]\s*(.*?)(?=\s*(?:\[|\*\*|^)?預期成效|$)", clean)
            outcome = re.search(r"(?:\[|\*\*|^)?預期成效(?:\]|\*\*)?[:：]\s*(.*)", clean)

            if not (target and action and outcome):
                return False, f"第 {i+1} 點格式錯誤 (找不到關鍵字: 目標對象/行動方案/預期成效)"

            try:
                items.append(
                    OptimizationItem(
                        target=target.group(1).strip(),
                        action=action.group(1).strip(),
                        outcome=outcome.group(1).strip()
                    )
                )
            except ValidationError as e:
                return False, f"第 {i+1} 點數據驗證失敗: {str(e)}"

        return True, items


# =========================
# Orchestrator
# =========================

class AgentOrchestrator:

    def __init__(self):
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.opt_agent = OptimizationAgent()

    async def run_flow(
        self,
        task: str,
        date_start: Optional[str],
        date_end: Optional[str],
    ) -> AgentRunResponse:

        steps: List[AgentStepResult] = []

        data_result = await self.data_agent.run(
            task=task,
            date_start=date_start,
            date_end=date_end
        )

        steps.append(
            AgentStepResult(
                name=self.data_agent.name,
                summary=str(data_result.get("summary"))[:200],
                raw_output=data_result,
            )
        )

        if not data_result.get("verified"):
            return AgentRunResponse(
                data_summary="任務失敗",
                analysis_insights="",
                optimization_suggestions="",
                steps=steps,
                verified=False
            )

        analysis_result = await self.analysis_agent.run(
            data_summary=data_result.get("summary")
        )

        steps.append(
            AgentStepResult(
                name=self.analysis_agent.name,
                summary=analysis_result.get("analysis", "")[:200],
                raw_output=analysis_result,
            )
        )

        # 🟢 融合版保留：正確傳遞 data_summary 給 OptimizationAgent
        opt_result = await self.opt_agent.run(
            analysis=analysis_result.get("analysis"),
            data_summary=data_result.get("summary") 
        )

        steps.append(
            AgentStepResult(
                name=self.opt_agent.name,
                summary=opt_result.get("suggestions", "")[:200],
                raw_output=opt_result,
            )
        )

        if not opt_result.get("verified"):
            return AgentRunResponse(
                data_summary=data_result.get("summary"),
                analysis_insights=analysis_result.get("analysis"),
                optimization_suggestions="優化建議生成失敗",
                structured_suggestions=[],
                quality_evaluation=opt_result.get("evaluation", {}),
                verified=False,
                steps=steps,
            )

        return AgentRunResponse(
            data_summary=data_result.get("summary"),
            analysis_insights=analysis_result.get("analysis"),
            optimization_suggestions=opt_result.get("suggestions"),
            structured_suggestions=opt_result.get("structured_data", []),
            quality_evaluation=opt_result.get("evaluation", {}),
            verified=True,
            steps=steps,
        )


_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
