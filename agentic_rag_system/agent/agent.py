"""
Main Pydantic AI agent for TikTok Shop agentic RAG with knowledge graph.
Specialized for TikTok Shop learning, product hunting, compliance, and strategy.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

from .prompts import TIKTOK_SYSTEM_PROMPT
from .providers import get_llm_model
from .models import AgentDependencies, TikTokCategory, ChunkResult, GraphSearchResult
from .db_utils import vector_search, hybrid_search, get_document, list_documents
from .graph_utils import search_knowledge_graph, get_entity_relationships

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# Initialize the TikTok Shop specialized agent
rag_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=TIKTOK_SYSTEM_PROMPT
)


# Tool implementations for TikTok Shop learning
@rag_agent.tool
async def tiktok_vector_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for TikTok Shop knowledge using semantic similarity.
    
    This tool performs vector similarity search across TikTok-related content
    including YouTube transcripts and Facebook group discussions.
    
    Args:
        query: Search query for TikTok Shop knowledge
        limit: Maximum number of results to return (1-20)
        category: Optional category filter (product_hunting, compliance, etc.)
    
    Returns:
        List of relevant TikTok Shop knowledge chunks
    """
    try:
        # Generate embedding for the query
        from .providers import get_embedding_client, get_embedding_model
        
        embedding_client = get_embedding_client()
        embedding_model = get_embedding_model()
        
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=f"TikTok Shop: {query}"  # Add context prefix
        )
        query_embedding = response.data[0].embedding
        
        # Perform vector search
        results = await vector_search(
            embedding=query_embedding,
            limit=limit
        )
        
        # Filter by category if specified
        if category:
            results = [r for r in results if category.lower() in r.get("metadata", {}).get("category", "").lower()]
        
        # Format for agent consumption
        formatted_results = []
        for r in results:
            formatted_results.append({
                "content": r["content"],
                "score": r["similarity"],
                "source": r["document_source"],
                "title": r["document_title"],
                "category": r.get("metadata", {}).get("category", "general"),
                "chunk_id": r["chunk_id"]
            })
        
        logger.info(f"TikTok vector search returned {len(formatted_results)} results for: {query}")
        return formatted_results
        
    except Exception as e:
        logger.error(f"TikTok vector search failed: {e}")
        return []


@rag_agent.tool
async def tiktok_graph_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    focus_area: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search TikTok Shop knowledge graph for relationships and insights.
    
    This tool queries the knowledge graph to find relationships between
    TikTok Shop concepts, strategies, and compliance requirements.
    
    Args:
        query: Search query for graph relationships
        focus_area: Optional focus area (product_hunting, compliance, etc.)
    
    Returns:
        List of knowledge graph facts and relationships
    """
    try:
        # Enhance query with TikTok context
        enhanced_query = f"TikTok Shop {focus_area or ''}: {query}".strip()
        
        results = await search_knowledge_graph(query=enhanced_query)
        
        # Format for agent consumption
        formatted_results = []
        for r in results:
            formatted_results.append({
                "fact": r["fact"],
                "confidence": r.get("confidence", 0.5),
                "temporal_info": r.get("valid_at"),
                "source_id": r.get("source_node_uuid", ""),
                "category": focus_area or "general"
            })
        
        logger.info(f"TikTok graph search returned {len(formatted_results)} results for: {query}")
        return formatted_results
        
    except Exception as e:
        logger.error(f"TikTok graph search failed: {e}")
        return []


@rag_agent.tool
async def tiktok_hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
    text_weight: float = 0.3,
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic and keyword matching for TikTok content.
    
    This tool combines vector similarity with keyword matching to find
    the most relevant TikTok Shop knowledge across all sources.
    
    Args:
        query: Search query for TikTok Shop knowledge
        limit: Maximum number of results to return (1-20)
        text_weight: Weight for text similarity vs vector similarity (0.0-1.0)
        category: Optional category filter
    
    Returns:
        List of TikTok Shop knowledge ranked by combined relevance
    """
    try:
        # Generate embedding for the query
        from .providers import get_embedding_client, get_embedding_model
        
        embedding_client = get_embedding_client()
        embedding_model = get_embedding_model()
        
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=f"TikTok Shop: {query}"
        )
        query_embedding = response.data[0].embedding
        
        # Perform hybrid search
        results = await hybrid_search(
            embedding=query_embedding,
            query_text=query,
            limit=limit,
            text_weight=text_weight
        )
        
        # Filter by category if specified
        if category:
            results = [r for r in results if category.lower() in r.get("metadata", {}).get("category", "").lower()]
        
        # Format for agent consumption
        formatted_results = []
        for r in results:
            formatted_results.append({
                "content": r["content"],
                "score": r["similarity"],
                "source": r["document_source"],
                "title": r["document_title"],
                "category": r.get("metadata", {}).get("category", "general"),
                "chunk_id": r["chunk_id"]
            })
        
        logger.info(f"TikTok hybrid search returned {len(formatted_results)} results for: {query}")
        return formatted_results
        
    except Exception as e:
        logger.error(f"TikTok hybrid search failed: {e}")
        return []


@rag_agent.tool
async def get_tiktok_strategy_insights(
    ctx: RunContext[AgentDependencies],
    strategy_type: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Get specific TikTok Shop strategy insights from the knowledge base.
    
    This tool retrieves targeted insights for specific TikTok Shop strategies
    like product hunting, viral content creation, or compliance management.
    
    Args:
        strategy_type: Type of strategy (product_hunting, viral_content, compliance, etc.)
        limit: Maximum number of insights to return
    
    Returns:
        List of strategy-specific insights and recommendations
    """
    try:
        # Search for strategy-specific content
        strategy_query = f"TikTok Shop {strategy_type} strategy best practices tips"
        
        from .providers import get_embedding_client, get_embedding_model
        
        embedding_client = get_embedding_client()
        embedding_model = get_embedding_model()
        
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=strategy_query
        )
        query_embedding = response.data[0].embedding
        
        # Get vector results
        vector_results = await vector_search(
            embedding=query_embedding,
            limit=limit
        )
        
        # Get graph insights
        graph_results = await search_knowledge_graph(query=strategy_query)
        
        # Combine and format results
        insights = []
        
        # Add vector-based insights
        for r in vector_results[:3]:
            insights.append({
                "type": "knowledge_content",
                "insight": r["content"][:300] + "...",
                "source": r["document_source"],
                "relevance": r["similarity"],
                "strategy_type": strategy_type
            })
        
        # Add graph-based insights
        for r in graph_results[:2]:
            insights.append({
                "type": "relationship_fact",
                "insight": r["fact"],
                "confidence": r.get("confidence", 0.5),
                "strategy_type": strategy_type
            })
        
        logger.info(f"Retrieved {len(insights)} strategy insights for: {strategy_type}")
        return insights
        
    except Exception as e:
        logger.error(f"Failed to get strategy insights: {e}")
        return []


@rag_agent.tool
async def get_tiktok_compliance_info(
    ctx: RunContext[AgentDependencies],
    compliance_topic: str
) -> List[Dict[str, Any]]:
    """
    Get TikTok Shop compliance information and guidelines.
    
    This tool retrieves specific compliance information for TikTok Shop
    including policies, violation handling, and reinstatement procedures.
    
    Args:
        compliance_topic: Specific compliance topic (violations, appeals, policies, etc.)
    
    Returns:
        List of compliance-related information and procedures
    """
    try:
        # Search for compliance-specific content
        compliance_query = f"TikTok Shop compliance {compliance_topic} policy guidelines violation appeal"
        
        # Use hybrid search for comprehensive results
        from .providers import get_embedding_client, get_embedding_model
        
        embedding_client = get_embedding_client()
        embedding_model = get_embedding_model()
        
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=compliance_query
        )
        query_embedding = response.data[0].embedding
        
        results = await hybrid_search(
            embedding=query_embedding,
            query_text=compliance_query,
            limit=8,
            text_weight=0.4  # Higher text weight for compliance terms
        )
        
        # Format compliance information
        compliance_info = []
        for r in results:
            compliance_info.append({
                "guideline": r["content"],
                "source": r["document_source"],
                "relevance": r["similarity"],
                "topic": compliance_topic,
                "document_title": r["document_title"]
            })
        
        logger.info(f"Retrieved {len(compliance_info)} compliance items for: {compliance_topic}")
        return compliance_info
        
    except Exception as e:
        logger.error(f"Failed to get compliance info: {e}")
        return []
