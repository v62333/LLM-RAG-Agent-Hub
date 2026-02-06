from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import pandas as pd
import re

from pydantic import ValidationError

from app.models.schemas import (
    AgentRunResponse, 
    AgentStepResult, 
    OptimizationItem,
    AgentTaskRequest 
)
from app.storage.file_storage import get_storage
from app.llm.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: str

    @abstractmethod
    async def run(self, **kwargs) -> Dict[str, Any]:
        ...


class DataAgent(BaseAgent):
    name = "DataAgent"

    async def run(self, **kwargs) -> Dict[str, Any]:
        # --- [驗證區塊 START] ---
        try:
            # 將傳入的 kwargs 直接映射到 AgentTaskRequest 模型
            # 這時候會自動觸發 check_task_not_empty 的檢查
            req = AgentTaskRequest(**kwargs)
            logger.info(f"[{self.name}] 參數驗證通過: {req.task}")

        except ValidationError as e:
            # 如果 task 為空，或者格式不符，這裡會捕捉到錯誤
            error_msg = e.errors()[0]['msg'] # 取出 "任務內容不能為空..."
            logger.error(f"[{self.name}] 驗證失敗: {error_msg}")
            
            # 回傳失敗狀態，這樣 Orchestrator 才知道這步掛了
            return {
                "summary": "數據讀取任務中止",
                "error": error_msg,
                "verified": False 
            }
        # --- [驗證區塊 END] ---


        # --- [執行區塊] ---
        try:
            storage = get_storage()
            csv_path = storage.get_ads_csv_path()
            logger.info(f"Loading ads data from {csv_path}")
            df = pd.read_csv(csv_path)

            # 2. 改用 req 物件來存取屬性 (原本是 kwargs.get)
            # 因為已經通過驗證，這裡的 req.date_start 肯定是安全的 (str 或 None)
            if req.date_start:
                df = df[df["date"] >= req.date_start]
            
            if req.date_end:
                df = df[df["date"] <= req.date_end]

            # --- 以下運算邏輯保持不變 ---
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
                "verified": True  # 標記執行成功
            }

        except Exception as e:
            logger.error(f"[{self.name}] 執行時發生錯誤: {e}")
            return {
                "summary": "系統執行錯誤",
                "error": str(e),
                "verified": False
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
        
        sys_prompt = (
            "你是一位資深成效型廣告優化專家，講求數據驅動與具體行動方案。\n"
            "【核心原則】\n"
            "1. 嚴禁給出模稜兩可的建議（如「請持續觀察」）。\n"
            "2. 你的建議必須包含：目標對象、行動方案、預期成效。\n"
            "3. 語氣必須專業、篤定，不要使用過多敬語。"
        )

        max_retries = 3
        final_markdown = ""
        last_error_hint = ""  # 用於存放反饋給 LLM 的錯誤訊息

        for attempt in range(max_retries):
            try:
                # 將 User Prompt 移入迴圈，實現動態錯誤反饋
                error_feedback = f"\n\n⚠️ 【上次輸出格式錯誤】\n原因：{last_error_hint}\n請務必針對此錯誤進行修正。" if last_error_hint else ""
                
                user_prompt = (
                    "### 輸出規範 (必須嚴格遵守)\n"
                        "1. 建議數量：請精確提供 3 到 5 點建議。\n"
                        "2. 輸出格式：必須使用 Markdown 編號清單（1., 2., 3. ...）。\n"
                        "3. 內容要求：每點建議需包含以下三個要素：\n"
                        "   - [目標對象]：具體的 Campaign 或受眾群體。\n"
                        "   - [行動方案]：調整預算、更換素材或優化出價的具體做法。\n"
                        "   - [預期成效]：預計能提升的指標（如：降低 CPA、提高 ROAS）。\n\n"
                    f"### 廣告表現分析數據\n'''\n{analysis_text}\n'''"
                    f"{error_feedback}"
                )

                current_temp = 0.3 + (attempt * 0.1)
                
                result = await client.generate(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    temperature=current_temp,
                    max_tokens=512,
                )
                final_markdown = result["output"]

                # 解構回傳值 (success, data_or_error)
                success, data_or_error = self._parse_and_validate_with_feedback(final_markdown)
                
                if success:
                    logger.info(f"[{self.name}] 驗證成功 (第 {attempt+1} 次)")
                    return {
                        "suggestions": final_markdown,
                        "structured_data": [item.model_dump() for item in data_or_error], 
                        "verified": True
                    }
                
                # 驗證失敗，更新提示訊息進入下一次重試
                last_error_hint = data_or_error
                logger.warning(f"[{self.name}] 驗證失敗 (第 {attempt+1} 次): {last_error_hint}")

            except Exception as e:
                logger.error(f"[{self.name}] 執行錯誤: {e}")

        logger.error(f"[{self.name}] 重試耗盡")
        return {
            "suggestions": final_markdown,
            "structured_data": [],
            "verified": False
        }

    def _parse_and_validate_with_feedback(self, text: str) -> Tuple[bool, Any]:
        items = []
        lines = re.findall(r"^\d+\.\s*(.*)", text, re.MULTILINE)
        
        if not lines:
            return False, "格式錯誤：未能偵測到編號清單 (例如 1. xxx)。"

        for i, line in enumerate(lines):
            target = re.search(r"\[目標對象\][:：]\s*(.*?)(?=\s*[|｜]|$)", line)
            action = re.search(r"\[行動方案\][:：]\s*(.*?)(?=\s*[|｜]|$)", line)
            outcome = re.search(r"\[預期成效\][:：]\s*(.*)", line)

            if not (target and action and outcome):
                return False, f"第 {i+1} 點標籤缺失，請確保包含 [目標對象]、[行動方案] 與 [預期成效]。"

            try:
                item = OptimizationItem(
                    target=target.group(1).strip(),
                    action=action.group(1).strip(),
                    outcome=outcome.group(1).strip()
                )
                items.append(item)
            except ValidationError as e:
                # 取得 Pydantic 具體的錯誤描述
                err = e.errors()[0]
                msg = err['msg']
                field = err['loc'][0]
                return False, f"第 {i+1} 點內容驗證失敗：欄位 '{field}' {msg}。"
        
        if len(items) < 3:
            return False, f"建議數量不足：目前僅解析出 {len(items)} 點，請提供至少 3 點。"

        return True, items


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