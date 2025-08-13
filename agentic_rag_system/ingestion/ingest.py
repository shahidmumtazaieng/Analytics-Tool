"""
Document ingestion pipeline for TikTok Shop Agentic RAG system.
Handles YouTube transcripts and Facebook group data processing.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from ..agent.models import IngestionConfig, TikTokKnowledgeSource
from ..agent.db_utils import db_pool, initialize_database
from ..agent.graph_utils import add_knowledge_to_graph, initialize_graph
from ..agent.providers import get_embedding_client, get_embedding_model
from .chunker import create_chunker, optimize_chunks_for_tiktok

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Enhanced ingestion pipeline for TikTok Shop content.
    Processes YouTube transcripts and Facebook group data.
    """
    
    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "tiktok_data",
        clean_before_ingest: bool = False
    ):
        """Initialize the ingestion pipeline."""
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        self.chunker = create_chunker(config)
        self.initialized = False
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "knowledge_graph_nodes": 0,
            "errors": []
        }
    
    async def initialize(self):
        """Initialize the ingestion pipeline."""
        if self.initialized:
            return
        
        try:
            # Initialize database and graph connections
            await initialize_database()
            await initialize_graph()
            
            # Create documents folder if it doesn't exist
            os.makedirs(self.documents_folder, exist_ok=True)
            
            self.initialized = True
            logger.info("Document ingestion pipeline initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ingestion pipeline: {e}")
            raise
    
    async def close(self):
        """Close the ingestion pipeline."""
        self.initialized = False
        logger.info("Document ingestion pipeline closed")
    
    async def ingest_youtube_transcripts(
        self,
        transcripts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest YouTube video transcripts.
        
        Args:
            transcripts: List of transcript data with metadata
        
        Returns:
            Ingestion results
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            results = []
            
            for transcript in transcripts:
                try:
                    # Create knowledge source
                    source = TikTokKnowledgeSource(
                        source_type="youtube_transcript",
                        source_id=transcript.get("video_id", str(uuid.uuid4())),
                        title=transcript.get("title", "Untitled Video"),
                        content=transcript.get("transcript", ""),
                        metadata={
                            "channel": transcript.get("channel", ""),
                            "duration": transcript.get("duration", 0),
                            "views": transcript.get("views", 0),
                            "upload_date": transcript.get("upload_date", ""),
                            "tags": transcript.get("tags", []),
                            "url": transcript.get("url", ""),
                            "source_type": "youtube_transcript"
                        },
                        created_at=datetime.now(),
                        category=self._categorize_content(
                            transcript.get("title", "") + " " + transcript.get("transcript", "")
                        )
                    )
                    
                    # Process the source
                    result = await self._process_knowledge_source(source)
                    results.append(result)
                    
                except Exception as e:
                    error_msg = f"Error processing YouTube transcript {transcript.get('video_id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    self.stats["errors"].append(error_msg)
            
            # Update statistics
            self.stats["total_documents"] += len(results)
            
            logger.info(f"Successfully processed {len(results)} YouTube transcripts")
            return {
                "processed_count": len(results),
                "results": results,
                "stats": self.stats
            }
            
        except Exception as e:
            logger.error(f"Error ingesting YouTube transcripts: {e}")
            raise
    
    async def ingest_facebook_data(
        self,
        facebook_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest Facebook groups posts and comments.
        
        Args:
            facebook_data: List of Facebook posts/comments with metadata
        
        Returns:
            Ingestion results
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            results = []
            
            for item in facebook_data:
                try:
                    source_type = "facebook_post" if item.get("type") == "post" else "facebook_comment"
                    
                    # Create knowledge source
                    source = TikTokKnowledgeSource(
                        source_type=source_type,
                        source_id=item.get("id", str(uuid.uuid4())),
                        title=item.get("title", f"{source_type.title()} - {item.get('author', 'Unknown')}"),
                        content=item.get("content", ""),
                        metadata={
                            "author": item.get("author", ""),
                            "group_name": item.get("group_name", ""),
                            "likes": item.get("likes", 0),
                            "comments_count": item.get("comments_count", 0),
                            "shares": item.get("shares", 0),
                            "post_date": item.get("post_date", ""),
                            "engagement_score": item.get("engagement_score", 0),
                            "url": item.get("url", ""),
                            "source_type": source_type
                        },
                        created_at=datetime.now(),
                        category=self._categorize_content(item.get("content", ""))
                    )
                    
                    # Process the source
                    result = await self._process_knowledge_source(source)
                    results.append(result)
                    
                except Exception as e:
                    error_msg = f"Error processing Facebook item {item.get('id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    self.stats["errors"].append(error_msg)
            
            # Update statistics
            self.stats["total_documents"] += len(results)
            
            logger.info(f"Successfully processed {len(results)} Facebook items")
            return {
                "processed_count": len(results),
                "results": results,
                "stats": self.stats
            }
            
        except Exception as e:
            logger.error(f"Error ingesting Facebook data: {e}")
            raise
    
    async def _process_knowledge_source(self, source: TikTokKnowledgeSource) -> Dict[str, Any]:
        """Process a single knowledge source through the pipeline."""
        try:
            # Step 1: Create document record
            document_id = await self._store_document(source)
            
            # Step 2: Chunk the content
            chunks = self.chunker.chunk_document(
                content=source.content,
                metadata=source.metadata
            )
            
            # Step 3: Optimize chunks for TikTok content
            optimized_chunks = optimize_chunks_for_tiktok(chunks)
            
            # Step 4: Generate embeddings and store chunks
            chunk_ids = await self._store_chunks(document_id, optimized_chunks)
            
            # Step 5: Add to knowledge graph
            graph_episode_id = await add_knowledge_to_graph(
                content=source.content,
                title=source.title,
                source=f"{source.source_type}:{source.source_id}",
                metadata=source.metadata
            )
            
            # Update statistics
            self.stats["total_chunks"] += len(chunk_ids)
            self.stats["total_embeddings"] += len(chunk_ids)
            if graph_episode_id != "error":
                self.stats["knowledge_graph_nodes"] += 1
            
            return {
                "document_id": document_id,
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
                "graph_episode_id": graph_episode_id,
                "category": source.category,
                "source_type": source.source_type
            }
            
        except Exception as e:
            logger.error(f"Error processing knowledge source: {e}")
            raise
    
    async def _store_document(self, source: TikTokKnowledgeSource) -> str:
        """Store document in the database."""
        try:
            async with db_pool.acquire() as conn:
                query = """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                """
                
                document_id = await conn.fetchval(
                    query,
                    source.title,
                    f"{source.source_type}:{source.source_id}",
                    source.content,
                    {
                        **source.metadata,
                        "category": source.category,
                        "source_type": source.source_type,
                        "created_at": source.created_at.isoformat()
                    }
                )
                
                return str(document_id)
                
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            raise
    
    async def _store_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Store chunks with embeddings in the database."""
        try:
            # Generate embeddings for all chunks
            embedding_client = get_embedding_client()
            embedding_model = get_embedding_model()
            
            chunk_texts = [chunk["content"] for chunk in chunks]
            
            # Generate embeddings in batch
            response = await embedding_client.embeddings.create(
                model=embedding_model,
                input=chunk_texts
            )
            
            embeddings = [data.embedding for data in response.data]
            
            # Store chunks with embeddings
            chunk_ids = []
            async with db_pool.acquire() as conn:
                for i, chunk in enumerate(chunks):
                    query = """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        RETURNING id
                    """
                    
                    chunk_id = await conn.fetchval(
                        query,
                        document_id,
                        chunk["content"],
                        embeddings[i],
                        chunk["chunk_index"],
                        chunk.get("metadata", {}),
                        chunk.get("metadata", {}).get("word_count", 0)
                    )
                    
                    chunk_ids.append(str(chunk_id))
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise
    
    def _categorize_content(self, content: str) -> str:
        """Categorize content into TikTok Shop focus areas."""
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
