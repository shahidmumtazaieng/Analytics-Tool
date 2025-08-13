"""
Advanced TikTok Shop Learning System with Agentic RAG and Knowledge Graph
Replaces traditional fine-tuning approach with dynamic agentic system
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

import asyncpg
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field

# Import agentic RAG components
from agentic_rag_system.agent.agent import rag_agent, AgentDependencies
from agentic_rag_system.agent.providers import get_llm_model, get_embedding_client, get_embedding_model
from agentic_rag_system.agent.db_utils import initialize_database, close_database, vector_search, hybrid_search
from agentic_rag_system.agent.graph_utils import initialize_graph, close_graph, search_knowledge_graph
from agentic_rag_system.agent.models import ChunkResult, GraphSearchResult
from agentic_rag_system.ingestion.ingest import DocumentIngestionPipeline
from agentic_rag_system.ingestion.chunker import ChunkingConfig

# Crawl4AI integration for live knowledge updates
from tiktok_knowledge_manager import TikTokKnowledgeManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TikTokKnowledgeSource:
    """TikTok knowledge source definition"""
    source_type: str  # 'youtube_transcript', 'facebook_post', 'facebook_comment'
    source_id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    category: str  # 'product_hunting', 'compliance', 'reinstatement', 'strategy', 'trends'

@dataclass
class AgenticRAGResponse:
    """Enhanced RAG response with agent information"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    tools_used: List[str]
    agent_reasoning: str
    knowledge_graph_insights: List[str]
    retrieval_time: float
    generation_time: float

class TikTokAgenticRAGSystem:
    """
    Advanced TikTok Shop Learning System using Agentic RAG with Knowledge Graph

    Features:
    - Multi-source data ingestion (YouTube transcripts, Facebook groups)
    - Pydantic AI agents for specialized tasks
    - PostgreSQL + pgvector for semantic search
    - Neo4j + Graphiti for knowledge relationships
    - Gemini API for cost-effective AI processing
    - Real-time agent orchestration
    """

    def __init__(self):
        """Initialize the agentic RAG system"""
        self.initialized = False
        self.agent_dependencies = None
        self.ingestion_pipeline = None
        self.knowledge_manager = None

        # Gemini configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        # Database configuration
        self.database_url = os.getenv("DATABASE_URL")
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        logger.info("TikTok Agentic RAG System initializing...")

    async def initialize(self):
        """Initialize all system components"""
        if self.initialized:
            return

        try:
            # Initialize database connections
            await initialize_database()
            await initialize_graph()

            # Set up agent dependencies for TikTok Shop learning
            self.agent_dependencies = AgentDependencies(
                session_id="tiktok_learning_session",
                user_id="tiktok_learner",
                search_preferences={
                    "use_vector": True,
                    "use_graph": True,
                    "default_limit": 10,
                    "focus_areas": ["product_hunting", "compliance", "reinstatement", "strategy", "trends"]
                }
            )

            # Initialize ingestion pipeline
            from agentic_rag_system.agent.models import IngestionConfig
            ingestion_config = IngestionConfig(
                chunk_size=1000,
                chunk_overlap=200,
                max_chunk_size=2000,
                use_semantic_chunking=True
            )

            self.ingestion_pipeline = DocumentIngestionPipeline(
                config=ingestion_config,
                documents_folder="tiktok_data",
                clean_before_ingest=False
            )

            await self.ingestion_pipeline.initialize()

            # Initialize Crawl4AI knowledge manager for live updates
            db_config = {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": os.getenv("POSTGRES_PORT", "5432"),
                "database": os.getenv("POSTGRES_DB", "tiktok_learning"),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", "")
            }
            self.knowledge_manager = TikTokKnowledgeManager(db_config)

            self.initialized = True
            logger.info("TikTok Agentic RAG System initialized successfully with Crawl4AI integration")

        except Exception as e:
            logger.error(f"Failed to initialize TikTok Agentic RAG System: {e}")
            raise

    async def update_live_knowledge(self, urls: List[str] = None) -> Dict[str, Any]:
        """Update knowledge base with live data from TikTok-related sources using Crawl4AI"""
        if not self.initialized or not self.knowledge_manager:
            raise RuntimeError("System not initialized")

        try:
            # Default TikTok knowledge sources if none provided
            if not urls:
                urls = [
                    "https://seller-us.tiktok.com/university",
                    "https://ads.tiktok.com/business/en-US/inspiration",
                    "https://www.tiktok.com/business/en-US/blog"
                ]

            logger.info(f"Updating knowledge base with live data from {len(urls)} sources")

            # Get stats before update
            stats_before = await self.knowledge_manager.get_knowledge_stats()

            # Update from sources using Crawl4AI
            await self.knowledge_manager.update_from_sources(urls)

            # Get stats after update
            stats_after = await self.knowledge_manager.get_knowledge_stats()

            # Calculate improvement
            new_chunks = stats_after.get('relevant_chunks', 0) - stats_before.get('relevant_chunks', 0)

            result = {
                "success": True,
                "new_chunks_added": new_chunks,
                "total_chunks": stats_after.get('relevant_chunks', 0),
                "sources_processed": len(urls),
                "updated_at": datetime.now().isoformat()
            }

            logger.info(f"Live knowledge update completed: {new_chunks} new chunks added")
            return result

        except Exception as e:
            logger.error(f"Error updating live knowledge: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }

    async def search_live_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search live knowledge base using Crawl4AI enhanced content"""
        if not self.initialized or not self.knowledge_manager:
            raise RuntimeError("System not initialized")

        try:
            # Search in live knowledge base
            live_results = await self.knowledge_manager.find_similar_content(query, limit=limit)

            # Format results for consistency with existing RAG system
            formatted_results = []
            for result in live_results:
                formatted_results.append({
                    "content": result["content"],
                    "source": result["source_url"],
                    "topics": result["topics"],
                    "relevance_score": result["relevance_score"],
                    "similarity": result["similarity"],
                    "metadata": result["metadata"],
                    "type": "live_knowledge"
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching live knowledge: {e}")
            return []

    async def close(self):
        """Close all system connections"""
        if self.ingestion_pipeline:
            await self.ingestion_pipeline.close()
        await close_graph()
        await close_database()
        self.initialized = False
    
    async def add_knowledge(self, knowledge_items: List[TikTokKnowledgeItem]):
        """Add knowledge items to Pinecone vector database"""
        try:
            vectors = []
            for item in knowledge_items:
                # Generate embedding
                embedding = self.embedding_model.encode(item.content).tolist()
                
                # Prepare metadata
                metadata = {
                    "title": item.title,
                    "content": item.content,
                    "category": item.category,
                    "tags": ",".join(item.tags),
                    "source": item.source,
                    "created_at": item.created_at.isoformat()
                }
                
                vectors.append({
                    "id": item.id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
            logger.info(f"Added {len(vectors)} knowledge items to Pinecone")
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            raise HTTPException(status_code=500, detail="Failed to add knowledge")

    async def ingest_youtube_transcripts(self, transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingest YouTube video transcripts into the system

        Args:
            transcripts: List of transcript data with metadata

        Returns:
            Ingestion results
        """
        try:
            knowledge_sources = []

            for transcript in transcripts:
                source = TikTokKnowledgeSource(
                    source_type="youtube_transcript",
                    source_id=transcript.get("video_id", ""),
                    title=transcript.get("title", ""),
                    content=transcript.get("transcript", ""),
                    metadata={
                        "channel": transcript.get("channel", ""),
                        "duration": transcript.get("duration", 0),
                        "views": transcript.get("views", 0),
                        "upload_date": transcript.get("upload_date", ""),
                        "tags": transcript.get("tags", [])
                    },
                    created_at=datetime.now(),
                    category=self._categorize_content(transcript.get("title", "") + " " + transcript.get("transcript", ""))
                )
                knowledge_sources.append(source)

            # Process through ingestion pipeline
            result = await self._process_knowledge_sources(knowledge_sources)

            logger.info(f"Successfully ingested {len(transcripts)} YouTube transcripts")
            return result

        except Exception as e:
            logger.error(f"Error ingesting YouTube transcripts: {e}")
            raise

    async def ingest_facebook_data(self, facebook_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingest Facebook groups posts and comments

        Args:
            facebook_data: List of Facebook posts/comments with metadata

        Returns:
            Ingestion results
        """
        try:
            knowledge_sources = []

            for item in facebook_data:
                source_type = "facebook_post" if item.get("type") == "post" else "facebook_comment"

                source = TikTokKnowledgeSource(
                    source_type=source_type,
                    source_id=item.get("id", ""),
                    title=item.get("title", f"{source_type.title()} - {item.get('author', 'Unknown')}"),
                    content=item.get("content", ""),
                    metadata={
                        "author": item.get("author", ""),
                        "group_name": item.get("group_name", ""),
                        "likes": item.get("likes", 0),
                        "comments_count": item.get("comments_count", 0),
                        "shares": item.get("shares", 0),
                        "post_date": item.get("post_date", ""),
                        "engagement_score": item.get("engagement_score", 0)
                    },
                    created_at=datetime.now(),
                    category=self._categorize_content(item.get("content", ""))
                )
                knowledge_sources.append(source)

            # Process through ingestion pipeline
            result = await self._process_knowledge_sources(knowledge_sources)

            logger.info(f"Successfully ingested {len(facebook_data)} Facebook items")
            return result

        except Exception as e:
            logger.error(f"Error ingesting Facebook data: {e}")
            raise
    
    async def handle_chat_request(self, message: str, user_id: str, rag_enabled: bool = True) -> AgenticRAGResponse:
        """
        Handle chat request using agentic RAG system

        Args:
            message: User's question/message
            user_id: User identifier
            rag_enabled: Whether to use RAG retrieval

        Returns:
            Enhanced response with agent insights
        """
        start_time = datetime.now()

        try:
            if not self.initialized:
                await self.initialize()

            # Determine the best approach based on the query
            query_category = self._categorize_query(message)
            tools_used = []

            # Step 1: Vector search for relevant content
            vector_results = []
            if rag_enabled:
                vector_start = datetime.now()

                # Generate embedding for the query
                embedding_client = get_embedding_client()
                embedding_model = get_embedding_model()

                response = await embedding_client.embeddings.create(
                    model=embedding_model,
                    input=message
                )
                query_embedding = response.data[0].embedding

                # Perform vector search
                vector_results = await vector_search(
                    embedding=query_embedding,
                    limit=10
                )
                tools_used.append("vector_search")

                vector_time = (datetime.now() - vector_start).total_seconds()
                logger.info(f"Vector search completed in {vector_time:.2f}s, found {len(vector_results)} results")

            # Step 2: Knowledge graph search for relationships
            graph_results = []
            if rag_enabled and self._should_use_graph(message):
                graph_start = datetime.now()

                graph_results = await search_knowledge_graph(query=message)
                tools_used.append("graph_search")

                graph_time = (datetime.now() - graph_start).total_seconds()
                logger.info(f"Graph search completed in {graph_time:.2f}s, found {len(graph_results)} results")

            # Step 3: Use Pydantic AI agent for intelligent response generation
            agent_start = datetime.now()

            # Prepare context for the agent
            context = {
                "query_category": query_category,
                "vector_results": [
                    {
                        "content": r["content"],
                        "source": r["document_source"],
                        "title": r["document_title"],
                        "score": r["similarity"]
                    }
                    for r in vector_results
                ],
                "graph_results": [
                    {
                        "fact": r["fact"],
                        "source": r.get("source_node_uuid", ""),
                        "temporal": r.get("valid_at", "")
                    }
                    for r in graph_results
                ],
                "focus_area": query_category
            }

            # Run the agent with TikTok-specific context
            agent_response = await self._run_tiktok_agent(message, context)
            tools_used.append("tiktok_specialized_agent")

            agent_time = (datetime.now() - agent_start).total_seconds()

            # Compile final response
            total_time = (datetime.now() - start_time).total_seconds()

            response = AgenticRAGResponse(
                answer=agent_response.get("answer", ""),
                sources=self._compile_sources(vector_results, graph_results),
                confidence=agent_response.get("confidence", 0.8),
                tools_used=tools_used,
                agent_reasoning=agent_response.get("reasoning", ""),
                knowledge_graph_insights=agent_response.get("graph_insights", []),
                retrieval_time=vector_time + graph_time if rag_enabled else 0.0,
                generation_time=agent_time
            )

            logger.info(f"TikTok Agentic RAG response generated in {total_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error in TikTok Agentic RAG chat: {e}")
            # Fallback response
            return AgenticRAGResponse(
                answer=f"I apologize, but I encountered an error processing your request about TikTok Shop. Please try rephrasing your question.",
                sources=[],
                confidence=0.0,
                tools_used=["error_fallback"],
                agent_reasoning="System error occurred",
                knowledge_graph_insights=[],
                retrieval_time=0.0,
                generation_time=0.0
            )

    def _categorize_content(self, content: str) -> str:
        """Categorize content into TikTok Shop focus areas"""
        content_lower = content.lower()

        if any(keyword in content_lower for keyword in ["product", "hunting", "research", "trending", "viral"]):
            return "product_hunting"
        elif any(keyword in content_lower for keyword in ["compliance", "policy", "violation", "guidelines"]):
            return "compliance"
        elif any(keyword in content_lower for keyword in ["reinstate", "appeal", "banned", "suspended", "restore"]):
            return "reinstatement"
        elif any(keyword in content_lower for keyword in ["strategy", "marketing", "growth", "optimization"]):
            return "strategy"
        elif any(keyword in content_lower for keyword in ["trend", "viral", "algorithm", "engagement"]):
            return "trends"
        else:
            return "general"

    def _categorize_query(self, query: str) -> str:
        """Categorize user query to determine best approach"""
        return self._categorize_content(query)

    def _should_use_graph(self, query: str) -> bool:
        """Determine if knowledge graph search would be beneficial"""
        graph_keywords = [
            "relationship", "connect", "how", "why", "cause", "effect",
            "strategy", "approach", "method", "process", "timeline"
        ]
        return any(keyword in query.lower() for keyword in graph_keywords)

    async def _process_knowledge_sources(self, sources: List[TikTokKnowledgeSource]) -> Dict[str, Any]:
        """Process knowledge sources through the ingestion pipeline"""
        try:
            # Convert to documents format for ingestion
            documents = []
            for source in sources:
                doc_content = f"Title: {source.title}\n\nContent: {source.content}\n\nCategory: {source.category}\n\nSource: {source.source_type}"

                # Create temporary file for ingestion
                doc_path = f"temp_{source.source_id}.md"
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(doc_content)

                documents.append({
                    "path": doc_path,
                    "title": source.title,
                    "metadata": source.metadata
                })

            # Run ingestion pipeline
            result = await self.ingestion_pipeline.ingest_documents()

            # Clean up temporary files
            for doc in documents:
                try:
                    os.remove(doc["path"])
                except:
                    pass

            return {
                "processed_sources": len(sources),
                "ingestion_result": result,
                "categories": [s.category for s in sources]
            }

        except Exception as e:
            logger.error(f"Error processing knowledge sources: {e}")
            raise

    async def _run_tiktok_agent(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the specialized TikTok agent with context"""
        try:
            # Use the agentic RAG agent with TikTok-specific prompt
            tiktok_prompt = f"""You are a specialized TikTok Shop learning assistant with expertise in:
            - Product hunting and trend analysis
            - TikTok Shop compliance and policies
            - Account reinstatement strategies
            - Marketing and growth strategies
            - Viral content creation

            Query Category: {context.get('query_category', 'general')}
            Focus Area: {context.get('focus_area', 'general')}

            Vector Search Results: {len(context.get('vector_results', []))} relevant documents found
            Knowledge Graph Results: {len(context.get('graph_results', []))} relationships found

            Based on the context provided, give a comprehensive, actionable response to help with TikTok Shop success.
            """

            # Prepare context for the agent
            vector_context = "\n\n".join([
                f"Source: {r['source']}\nTitle: {r['title']}\nContent: {r['content']}\nRelevance: {r['score']:.2f}"
                for r in context.get('vector_results', [])[:5]
            ])

            graph_context = "\n\n".join([
                f"Fact: {r['fact']}\nSource: {r['source']}\nTemporal: {r['temporal']}"
                for r in context.get('graph_results', [])[:3]
            ])

            full_context = f"{tiktok_prompt}\n\nRelevant Information:\n{vector_context}\n\nRelated Facts:\n{graph_context}\n\nUser Question: {message}"

            # Use Gemini for response generation
            gemini_client = get_llm_model()  # This should be configured for Gemini

            # For now, return a structured response
            # TODO: Integrate with actual Pydantic AI agent

            return {
                "answer": f"Based on the available TikTok Shop knowledge and your question about '{message}', here's my analysis:\n\n[This would be the agent's response using the context and specialized TikTok knowledge]",
                "confidence": 0.85,
                "reasoning": f"Used {len(context.get('vector_results', []))} vector results and {len(context.get('graph_results', []))} graph insights for category: {context.get('query_category', 'general')}",
                "graph_insights": [f"Found {len(context.get('graph_results', []))} related facts and relationships"]
            }

        except Exception as e:
            logger.error(f"Error running TikTok agent: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your TikTok Shop question. Please try rephrasing your question.",
                "confidence": 0.0,
                "reasoning": "Agent error occurred",
                "graph_insights": []
            }

    def _compile_sources(self, vector_results: List[Dict], graph_results: List[Dict]) -> List[Dict[str, Any]]:
        """Compile sources from vector and graph results"""
        sources = []

        # Add vector sources
        for result in vector_results[:5]:
            sources.append({
                "type": "document",
                "title": result.get("document_title", ""),
                "source": result.get("document_source", ""),
                "relevance_score": result.get("similarity", 0.0),
                "content_preview": result.get("content", "")[:200] + "..."
            })

        # Add graph sources
        for result in graph_results[:3]:
            sources.append({
                "type": "knowledge_graph",
                "fact": result.get("fact", ""),
                "source_id": result.get("source_node_uuid", ""),
                "temporal_info": result.get("valid_at", ""),
                "relevance_score": 0.8  # Default for graph results
            })

        return sources


# Service class for FastAPI integration
class TikTokLearningService:
    """Enhanced TikTok Learning Service with Agentic RAG"""

    def __init__(self):
        """Initialize the agentic RAG service"""
        self.agentic_rag = TikTokAgenticRAGSystem()
        self._initialized = False

    async def initialize(self):
        """Initialize the service"""
        if not self._initialized:
            await self.agentic_rag.initialize()
            self._initialized = True

    async def handle_chat_request(self, message: str, user_id: str, rag_enabled: bool = True) -> Dict[str, Any]:
        """Handle chat request with enhanced agentic RAG"""
        try:
            if not self._initialized:
                await self.initialize()

            if rag_enabled:
                # Use the new agentic RAG system
                response = await self.agentic_rag.handle_chat_request(message, user_id, rag_enabled)

                return {
                    "response": response.answer,
                    "sources": response.sources,
                    "confidence": response.confidence,
                    "model_used": "TikTok-Agentic-RAG-v2.0",
                    "tools_used": response.tools_used,
                    "agent_reasoning": response.agent_reasoning,
                    "knowledge_graph_insights": response.knowledge_graph_insights,
                    "retrieval_time": response.retrieval_time,
                    "generation_time": response.generation_time
                }
            else:
                # Simple response without RAG
                return {
                    "response": "I'm a TikTok Shop learning assistant. Please enable RAG for enhanced responses with access to YouTube transcripts and Facebook group insights.",
                    "sources": [],
                    "confidence": 0.5,
                    "model_used": "TikTok-Simple-v1.0",
                    "tools_used": ["simple_response"],
                    "agent_reasoning": "RAG disabled by user",
                    "knowledge_graph_insights": [],
                    "retrieval_time": 0.0,
                    "generation_time": 0.0
                }

        except Exception as e:
            logger.error(f"Error in TikTok Learning Service: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "sources": [],
                "confidence": 0.0,
                "model_used": "error",
                "tools_used": ["error_handler"],
                "agent_reasoning": "Service error occurred",
                "knowledge_graph_insights": [],
                "retrieval_time": 0.0,
                "generation_time": 0.0
            }

    async def ingest_youtube_data(self, transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest YouTube transcript data"""
        if not self._initialized:
            await self.initialize()
        return await self.agentic_rag.ingest_youtube_transcripts(transcripts)

    async def ingest_facebook_data(self, facebook_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest Facebook groups data"""
        if not self._initialized:
            await self.initialize()
        return await self.agentic_rag.ingest_facebook_data(facebook_data)

</xaiArtifact> 