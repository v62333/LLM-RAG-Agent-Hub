from typing import List, Optional, Any
from pydantic import BaseModel
from .enums import Domain, CollectionName
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List


class HealthResponse(BaseModel):
    status: str = "ok"


class PromptRequest(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt: str
    domain: Domain = Domain.general
    temperature: float = 0.2
    max_tokens: int = 512


class PromptResponse(BaseModel):
    output: str
    model: str
    usage: Optional[dict] = None


class EmbedRequest(BaseModel):
    texts: List[str]
    collection: CollectionName = CollectionName.custom
    store: bool = False


class EmbedResult(BaseModel):
    text: str
    vector_id: Optional[str] = None
    score: Optional[float] = None


class EmbedResponse(BaseModel):
    results: List[EmbedResult]


class IngestDocsRequest(BaseModel):
    file_paths: List[str]
    collection: CollectionName = CollectionName.docs
    overwrite: bool = False


class IngestDocsResponse(BaseModel):
    success_count: int
    failed_files: List[str]


class RagAskRequest(BaseModel):
    question: str
    top_k: int = 5
    collection: CollectionName = CollectionName.docs
    use_hybrid: bool = False


class SourceChunk(BaseModel):
    doc_id: str
    doc_name: Optional[str] = None
    chunk_id: int
    score: float
    snippet: str


class RagAnswer(BaseModel):
    answer: str
    strategy: str
    sources: List[SourceChunk]
    metadata: Optional[dict] = None


class RagAskResponse(BaseModel):
    result: RagAnswer


class RecommendNewsRequest(BaseModel):
    recent_queries: List[str]
    preferred_tags: Optional[List[str]] = None
    top_k: int = 5


class NewsItem(BaseModel):
    id: str
    title: str
    content_snippet: str
    tags: List[str]
    published_at: str
    score: float


class RecommendNewsResponse(BaseModel):
    items: List[NewsItem]


class AgentTaskRequest(BaseModel):
    task: str
    date_start: Optional[str] = None
    date_end: Optional[str] = None

    @field_validator('task')
    @classmethod
    def check_task_not_empty(cls, v: str):
        # 去除前後空白後，檢查是否為空
        if not v.strip():
            raise ValueError("任務內容 (task) 不能為空或僅包含空白")
        return v
    
    # (選用) 如果您希望當前端傳送 "" (空字串) 作為日期時，自動轉成 None，可以加這段
    # 這樣 DataAgent 裡的 `if date_start:` 判斷會更準確
    @field_validator('date_start', 'date_end')
    @classmethod
    def empty_string_to_none(cls, v: Optional[str]):
        if v is not None and not v.strip():
            return None
        return v


class AgentStepResult(BaseModel):
    name: str
    summary: str
    raw_output: Optional[Any] = None


class AgentRunResponse(BaseModel):
    data_summary: str
    analysis_insights: str
    optimization_suggestions: str
    steps: List[AgentStepResult]

class OptimizationItem(BaseModel):
    """
    單一優化建議的結構限制
    """
    target: str = Field(
        ..., 
        description="目標對象，例如 'Campaign A' 或 '25-34歲男性'", 
        min_length=2
    )
    action: str = Field(
        ..., 
        description="具體行動方案，必須包含動作與數值調整", 
        min_length=5
    )
    outcome: str = Field(
        ..., 
        description="預期成效，需包含指標名稱", 
        min_length=2
    )

    @field_validator('target', 'action', 'outcome')
    @classmethod
    def check_not_empty_or_meaningless(cls, v: str):
        v = v.strip()
        invalid_keywords = ["無", "n/a", "未知", "none", "unknown"]
        
        if v.lower() in invalid_keywords:
            raise ValueError(f"欄位內容無效: '{v}'，請提供具體建議")
        return v
