"""
Pydantic models for data validation and serialization.
Enhanced for TikTok Shop learning with agentic capabilities.
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SearchType(str, Enum):
    """Search type enumeration."""
    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH = "graph"
    AGENTIC = "agentic"


class TikTokCategory(str, Enum):
    """TikTok Shop focus categories."""
    PRODUCT_HUNTING = "product_hunting"
    COMPLIANCE = "compliance"
    REINSTATEMENT = "reinstatement"
    STRATEGY = "strategy"
    TRENDS = "trends"
    GENERAL = "general"


# Enhanced Response Models
class ChunkResult(BaseModel):
    """Chunk search result model."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_title: str
    document_source: str
    category: Optional[TikTokCategory] = None
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is between 0 and 1."""
        return max(0.0, min(1.0, v))


class GraphSearchResult(BaseModel):
    """Knowledge graph search result model."""
    fact: str
    uuid: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    source_node_uuid: Optional[str] = None
    confidence: Optional[float] = None
    category: Optional[TikTokCategory] = None


class AgenticRAGResponse(BaseModel):
    """Enhanced RAG response with agent information."""
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    tools_used: List[str] = Field(default_factory=list)
    agent_reasoning: str = ""
    knowledge_graph_insights: List[str] = Field(default_factory=list)
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    category: Optional[TikTokCategory] = None
    model_used: str = "TikTok-Agentic-RAG-v2.0"


class TikTokKnowledgeSource(BaseModel):
    """TikTok knowledge source definition."""
    source_type: Literal["youtube_transcript", "facebook_post", "facebook_comment"]
    source_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    category: TikTokCategory


# Request Models
class ChatRequest(BaseModel):
    """Enhanced chat request model."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    search_type: SearchType = Field(default=SearchType.AGENTIC, description="Type of search to perform")
    rag_enabled: bool = Field(default=True, description="Enable RAG retrieval")
    use_knowledge_graph: bool = Field(default=True, description="Use knowledge graph search")
    focus_category: Optional[TikTokCategory] = Field(None, description="Focus on specific category")
    
    model_config = ConfigDict(use_enum_values=True)


class IngestionRequest(BaseModel):
    """Data ingestion request model."""
    source_type: Literal["youtube", "facebook"]
    data: List[Dict[str, Any]]
    batch_size: int = Field(default=10, ge=1, le=100)
    clean_before_ingest: bool = Field(default=False)


# Configuration Models
class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=8000)
    use_semantic_chunking: bool = Field(default=True)
    embedding_model: str = Field(default="text-embedding-004")
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v


class AgentDependencies(BaseModel):
    """Dependencies for the agent."""
    session_id: str
    user_id: Optional[str] = None
    search_preferences: Dict[str, Any] = Field(default_factory=dict)
    focus_areas: List[TikTokCategory] = Field(default_factory=list)
    
    def __post_init__(self):
        if not self.search_preferences:
            self.search_preferences = {
                "use_vector": True,
                "use_graph": True,
                "default_limit": 10,
                "focus_areas": ["product_hunting", "compliance", "reinstatement", "strategy", "trends"]
            }
        
        if not self.focus_areas:
            self.focus_areas = [
                TikTokCategory.PRODUCT_HUNTING,
                TikTokCategory.COMPLIANCE,
                TikTokCategory.REINSTATEMENT,
                TikTokCategory.STRATEGY,
                TikTokCategory.TRENDS
            ]


# Tool Models
class ToolCall(BaseModel):
    """Tool call tracking model."""
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class SearchResponse(BaseModel):
    """Enhanced search response model."""
    results: List[ChunkResult] = Field(default_factory=list)
    graph_results: List[GraphSearchResult] = Field(default_factory=list)
    total_results: int = 0
    search_type: SearchType
    query_time_ms: float
    tools_used: List[ToolCall] = Field(default_factory=list)
    agent_insights: List[str] = Field(default_factory=list)


# Document Models
class DocumentMetadata(BaseModel):
    """Enhanced document metadata model."""
    id: str
    title: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None
    category: Optional[TikTokCategory] = None
    source_type: Optional[str] = None


class IngestionResult(BaseModel):
    """Result of document ingestion process."""
    total_documents: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    knowledge_graph_nodes: int = 0
    knowledge_graph_relationships: int = 0
    processing_time: float = 0.0
    errors: List[str] = Field(default_factory=list)
    categories_processed: List[TikTokCategory] = Field(default_factory=list)


# Health and Status Models
class HealthStatus(BaseModel):
    """System health status model."""
    status: Literal["healthy", "degraded", "unhealthy"]
    database_connected: bool = False
    graph_database_connected: bool = False
    embedding_service_available: bool = False
    llm_service_available: bool = False
    last_check: datetime
    version: str = "2.0.0"


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


# Streaming Models
class StreamDelta(BaseModel):
    """Streaming response delta."""
    content: str = ""
    tools_used: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    finished: bool = False
    error: Optional[str] = None
