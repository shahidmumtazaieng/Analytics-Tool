"""
Agentic RAG System for TikTok Shop Learning
Integrated into the Backend for enhanced product research capabilities
"""

__version__ = "2.0.0"
__author__ = "TikTok Shop Learning Team"

from .agent.agent import rag_agent, AgentDependencies
from .agent.providers import get_llm_model, get_embedding_client
from .agent.models import ChunkResult, GraphSearchResult, AgenticRAGResponse
from .ingestion.ingest import DocumentIngestionPipeline

__all__ = [
    "rag_agent",
    "AgentDependencies", 
    "get_llm_model",
    "get_embedding_client",
    "ChunkResult",
    "GraphSearchResult", 
    "AgenticRAGResponse",
    "DocumentIngestionPipeline"
]
