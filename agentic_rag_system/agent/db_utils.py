"""
Database utilities for PostgreSQL with pgvector.
Simplified version for TikTok Shop learning integration.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global connection pool
db_pool: Optional[asyncpg.Pool] = None


async def initialize_database():
    """Initialize database connection pool."""
    global db_pool
    
    if db_pool is not None:
        return
    
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        db_pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=10,
            command_timeout=60
        )
        
        logger.info("Database connection pool initialized")
        
        # Test connection
        async with db_pool.acquire() as conn:
            await conn.execute("SELECT 1")
        
        logger.info("Database connection test successful")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_database():
    """Close database connection pool."""
    global db_pool
    
    if db_pool is not None:
        await db_pool.close()
        db_pool = None
        logger.info("Database connection pool closed")


async def test_connection() -> bool:
    """Test database connection."""
    try:
        if db_pool is None:
            await initialize_database()
        
        async with db_pool.acquire() as conn:
            await conn.execute("SELECT 1")
        
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def vector_search(
    embedding: List[float],
    limit: int = 10,
    similarity_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search.
    
    Args:
        embedding: Query embedding vector
        limit: Maximum number of results
        similarity_threshold: Minimum similarity score
    
    Returns:
        List of matching chunks with metadata
    """
    try:
        if db_pool is None:
            await initialize_database()
        
        async with db_pool.acquire() as conn:
            # Use the match_chunks function from schema
            query = """
                SELECT 
                    chunk_id,
                    document_id,
                    content,
                    similarity,
                    metadata,
                    document_title,
                    document_source
                FROM match_chunks($1::vector, $2::int)
                WHERE similarity >= $3
                ORDER BY similarity DESC
            """
            
            rows = await conn.fetch(query, embedding, limit, similarity_threshold)
            
            results = []
            for row in rows:
                results.append({
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "similarity": float(row["similarity"]),
                    "metadata": row["metadata"] or {},
                    "document_title": row["document_title"],
                    "document_source": row["document_source"]
                })
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining vector similarity and text search.
    
    Args:
        embedding: Query embedding vector
        query_text: Text query for keyword search
        limit: Maximum number of results
        text_weight: Weight for text similarity (0-1)
    
    Returns:
        List of matching chunks with combined scores
    """
    try:
        if db_pool is None:
            await initialize_database()
        
        async with db_pool.acquire() as conn:
            # Use the hybrid_search function from schema
            query = """
                SELECT 
                    chunk_id,
                    document_id,
                    content,
                    combined_score as similarity,
                    metadata,
                    document_title,
                    document_source
                FROM hybrid_search($1::vector, $2::text, $3::int, $4::float)
                ORDER BY combined_score DESC
            """
            
            rows = await conn.fetch(query, embedding, query_text, limit, text_weight)
            
            results = []
            for row in rows:
                results.append({
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "similarity": float(row["similarity"]),
                    "metadata": row["metadata"] or {},
                    "document_title": row["document_title"],
                    "document_source": row["document_source"]
                })
            
            logger.info(f"Hybrid search returned {len(results)} results")
            return results
            
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []


async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a complete document by ID.
    
    Args:
        document_id: Document UUID
    
    Returns:
        Document data or None if not found
    """
    try:
        if db_pool is None:
            await initialize_database()
        
        async with db_pool.acquire() as conn:
            # Get document
            doc_query = """
                SELECT id, title, source, content, metadata, created_at, updated_at
                FROM documents 
                WHERE id = $1
            """
            
            doc_row = await conn.fetchrow(doc_query, document_id)
            if not doc_row:
                return None
            
            # Get chunks
            chunks_query = """
                SELECT id, content, chunk_index, metadata, token_count
                FROM chunks 
                WHERE document_id = $1
                ORDER BY chunk_index
            """
            
            chunk_rows = await conn.fetch(chunks_query, document_id)
            
            return {
                "id": str(doc_row["id"]),
                "title": doc_row["title"],
                "source": doc_row["source"],
                "content": doc_row["content"],
                "metadata": doc_row["metadata"] or {},
                "created_at": doc_row["created_at"].isoformat(),
                "updated_at": doc_row["updated_at"].isoformat(),
                "chunks": [
                    {
                        "id": str(chunk["id"]),
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "metadata": chunk["metadata"] or {},
                        "token_count": chunk["token_count"]
                    }
                    for chunk in chunk_rows
                ]
            }
            
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        return None


async def list_documents(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    """
    List documents with metadata.
    
    Args:
        limit: Maximum number of documents
        offset: Number of documents to skip
    
    Returns:
        List of document metadata
    """
    try:
        if db_pool is None:
            await initialize_database()
        
        async with db_pool.acquire() as conn:
            query = """
                SELECT 
                    d.id,
                    d.title,
                    d.source,
                    d.metadata,
                    d.created_at,
                    d.updated_at,
                    COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
                ORDER BY d.created_at DESC
                LIMIT $1 OFFSET $2
            """
            
            rows = await conn.fetch(query, limit, offset)
            
            return [
                {
                    "id": str(row["id"]),
                    "title": row["title"],
                    "source": row["source"],
                    "metadata": row["metadata"] or {},
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "chunk_count": row["chunk_count"]
                }
                for row in rows
            ]
            
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return []


async def store_chat_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Store a chat message.
    
    Args:
        session_id: Session UUID
        user_id: User identifier
        role: Message role (user/assistant/system)
        content: Message content
        metadata: Additional metadata
    
    Returns:
        Message ID
    """
    try:
        if db_pool is None:
            await initialize_database()
        
        async with db_pool.acquire() as conn:
            # Ensure session exists
            session_query = """
                INSERT INTO sessions (id, user_id, metadata)
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            """
            await conn.execute(session_query, session_id, user_id, metadata or {})
            
            # Insert message
            message_query = """
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """
            
            message_id = await conn.fetchval(
                message_query, session_id, role, content, metadata or {}
            )
            
            return str(message_id)
            
    except Exception as e:
        logger.error(f"Failed to store chat message: {e}")
        raise
