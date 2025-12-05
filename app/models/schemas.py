from typing import List, Optional, Any
from pydantic import BaseModel
from .enums import Domain, CollectionName


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


class AgentStepResult(BaseModel):
    name: str
    summary: str
    raw_output: Optional[Any] = None


class AgentRunResponse(BaseModel):
    data_summary: str
    analysis_insights: str
    optimization_suggestions: str
    steps: List[AgentStepResult]
