"""
tiktok_knowledge_manager.py
---------------------------
Management utilities for TikTok knowledge base with Crawl4AI integration.
Provides functions to update, query, and maintain the TikTok learning knowledge base.

Features:
- Scheduled knowledge updates
- Content quality assessment
- Duplicate detection and removal
- Knowledge base statistics
- Integration with existing agentic RAG system
"""

import asyncio
import asyncpg
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from tiktok_knowledge_crawler import TikTokKnowledgeCrawler

class TikTokKnowledgeManager:
    """Manager for TikTok knowledge base operations"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.crawler = TikTokKnowledgeCrawler(db_config)
    
    async def get_knowledge_stats(self, table_name: str = "tiktok_knowledge") -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            conn = await asyncpg.connect(**self.db_config)
            
            # Basic stats
            total_chunks = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            relevant_chunks = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name} WHERE is_tiktok_relevant = true")
            
            # Topic distribution
            topic_stats = await conn.fetch(f"""
                SELECT unnest(topics) as topic, COUNT(*) as count
                FROM {table_name}
                WHERE is_tiktok_relevant = true
                GROUP BY topic
                ORDER BY count DESC
                LIMIT 10
            """)
            
            # Relevance score distribution
            relevance_stats = await conn.fetch(f"""
                SELECT relevance_score, COUNT(*) as count
                FROM {table_name}
                WHERE is_tiktok_relevant = true
                GROUP BY relevance_score
                ORDER BY relevance_score DESC
            """)
            
            # Recent additions
            recent_count = await conn.fetchval(f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE created_at > NOW() - INTERVAL '7 days'
            """)
            
            # Source distribution
            source_stats = await conn.fetch(f"""
                SELECT source_url, COUNT(*) as chunk_count
                FROM {table_name}
                WHERE is_tiktok_relevant = true
                GROUP BY source_url
                ORDER BY chunk_count DESC
                LIMIT 10
            """)
            
            await conn.close()
            
            return {
                "total_chunks": total_chunks,
                "relevant_chunks": relevant_chunks,
                "relevance_percentage": (relevant_chunks / total_chunks * 100) if total_chunks > 0 else 0,
                "recent_additions": recent_count,
                "top_topics": [{"topic": row["topic"], "count": row["count"]} for row in topic_stats],
                "relevance_distribution": [{"score": row["relevance_score"], "count": row["count"]} for row in relevance_stats],
                "top_sources": [{"url": row["source_url"], "chunks": row["chunk_count"]} for row in source_stats],
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting knowledge stats: {str(e)}")
            return {}
    
    async def find_similar_content(self, query: str, table_name: str = "tiktok_knowledge", limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar content using vector similarity"""
        try:
            conn = await asyncpg.connect(**self.db_config)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Find similar content
            results = await conn.fetch(f"""
                SELECT content, metadata, source_url, topics, relevance_score,
                       1 - (embedding <=> $1::vector) as similarity
                FROM {table_name}
                WHERE is_tiktok_relevant = true
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """, query_embedding, limit)
            
            await conn.close()
            
            return [
                {
                    "content": row["content"],
                    "metadata": row["metadata"],
                    "source_url": row["source_url"],
                    "topics": row["topics"],
                    "relevance_score": row["relevance_score"],
                    "similarity": float(row["similarity"])
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"Error finding similar content: {str(e)}")
            return []
    
    async def remove_duplicates(self, table_name: str = "tiktok_knowledge", similarity_threshold: float = 0.95):
        """Remove duplicate content based on embedding similarity"""
        try:
            conn = await asyncpg.connect(**self.db_config)
            
            # Get all embeddings and IDs
            rows = await conn.fetch(f"""
                SELECT id, embedding, content
                FROM {table_name}
                WHERE is_tiktok_relevant = true
                ORDER BY id
            """)
            
            if len(rows) < 2:
                print("Not enough content to check for duplicates")
                return
            
            # Convert embeddings to numpy array
            embeddings = np.array([row["embedding"] for row in rows])
            ids = [row["id"] for row in rows]
            
            # Calculate similarity matrix
            similarities = np.dot(embeddings, embeddings.T)
            
            # Find duplicates
            duplicates_to_remove = set()
            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    if similarities[i][j] > similarity_threshold:
                        # Keep the one with higher ID (more recent)
                        duplicates_to_remove.add(min(ids[i], ids[j]))
            
            if duplicates_to_remove:
                # Remove duplicates
                await conn.execute(f"""
                    DELETE FROM {table_name}
                    WHERE id = ANY($1::int[])
                """, list(duplicates_to_remove))
                
                print(f"Removed {len(duplicates_to_remove)} duplicate entries")
            else:
                print("No duplicates found")
            
            await conn.close()
            
        except Exception as e:
            print(f"Error removing duplicates: {str(e)}")
    
    async def update_from_sources(self, sources: List[str], table_name: str = "tiktok_knowledge"):
        """Update knowledge base from multiple sources"""
        print(f"Updating knowledge base from {len(sources)} sources...")
        
        total_new_chunks = 0
        
        for source in sources:
            try:
                print(f"Processing: {source}")
                
                # Crawl the source
                if source.endswith('sitemap.xml') or 'sitemap' in source:
                    results = await self.crawler.crawl_sitemap_tiktok_focused(source, max_urls=50)
                else:
                    results = await self.crawler.crawl_with_tiktok_focus(source)
                
                # Process results
                all_chunks = []
                for result in results:
                    if result['success'] and result['markdown']:
                        chunks = self.crawler.smart_chunk_tiktok_content(result['markdown'])
                        
                        for chunk in chunks:
                            chunk['source_url'] = result['url']
                            chunk['crawled_at'] = result.get('crawled_at', datetime.now().isoformat())
                        
                        relevant_chunks = [chunk for chunk in chunks if chunk['is_tiktok_relevant']]
                        all_chunks.extend(relevant_chunks)
                
                if all_chunks:
                    await self.crawler.insert_into_postgres(all_chunks, table_name=table_name)
                    total_new_chunks += len(all_chunks)
                    print(f"Added {len(all_chunks)} chunks from {source}")
                else:
                    print(f"No relevant content found in {source}")
                
                # Add delay between sources
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error processing {source}: {str(e)}")
                continue
        
        print(f"Knowledge base update complete. Added {total_new_chunks} new chunks.")
        
        # Remove duplicates after update
        await self.remove_duplicates(table_name)
    
    async def cleanup_old_content(self, table_name: str = "tiktok_knowledge", days_old: int = 90):
        """Remove content older than specified days"""
        try:
            conn = await asyncpg.connect(**self.db_config)
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            deleted_count = await conn.fetchval(f"""
                DELETE FROM {table_name}
                WHERE created_at < $1
                RETURNING COUNT(*)
            """, cutoff_date)
            
            await conn.close()
            
            print(f"Removed {deleted_count} entries older than {days_old} days")
            
        except Exception as e:
            print(f"Error cleaning up old content: {str(e)}")
    
    async def export_knowledge_base(self, table_name: str = "tiktok_knowledge", output_file: str = "tiktok_knowledge_export.json"):
        """Export knowledge base to JSON file"""
        try:
            conn = await asyncpg.connect(**self.db_config)
            
            rows = await conn.fetch(f"""
                SELECT content, metadata, source_url, topics, relevance_score, created_at
                FROM {table_name}
                WHERE is_tiktok_relevant = true
                ORDER BY created_at DESC
            """)
            
            await conn.close()
            
            export_data = [
                {
                    "content": row["content"],
                    "metadata": row["metadata"],
                    "source_url": row["source_url"],
                    "topics": row["topics"],
                    "relevance_score": row["relevance_score"],
                    "created_at": row["created_at"].isoformat()
                }
                for row in rows
            ]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"Exported {len(export_data)} entries to {output_file}")
            
        except Exception as e:
            print(f"Error exporting knowledge base: {str(e)}")

# Predefined TikTok Shop knowledge sources
TIKTOK_KNOWLEDGE_SOURCES = [
    # Official TikTok resources
    "https://seller-us.tiktok.com/university",
    "https://ads.tiktok.com/business/en-US/inspiration",
    
    # E-commerce and social commerce blogs
    "https://blog.hootsuite.com/tiktok-marketing/",
    "https://www.shopify.com/blog/tiktok-marketing",
    "https://blog.hubspot.com/marketing/tiktok-marketing",
    
    # Industry publications
    "https://www.socialmediaexaminer.com/tag/tiktok/",
    "https://sproutsocial.com/insights/tiktok-marketing/",
    
    # Add your own sources here
]

async def scheduled_update():
    """Scheduled update function for knowledge base"""
    db_config = {
        "host": "localhost",
        "port": "5432",
        "database": "tiktok_learning",
        "user": "postgres",
        "password": ""  # Set your password
    }
    
    manager = TikTokKnowledgeManager(db_config)
    
    print("Starting scheduled TikTok knowledge base update...")
    
    # Get current stats
    stats_before = await manager.get_knowledge_stats()
    print(f"Current knowledge base: {stats_before.get('relevant_chunks', 0)} relevant chunks")
    
    # Update from sources
    await manager.update_from_sources(TIKTOK_KNOWLEDGE_SOURCES)
    
    # Get updated stats
    stats_after = await manager.get_knowledge_stats()
    print(f"Updated knowledge base: {stats_after.get('relevant_chunks', 0)} relevant chunks")
    
    # Cleanup old content (optional)
    # await manager.cleanup_old_content(days_old=90)
    
    print("Scheduled update complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TikTok Knowledge Base Manager")
    parser.add_argument("action", choices=["update", "stats", "cleanup", "export", "find"], help="Action to perform")
    parser.add_argument("--query", help="Query for find action")
    parser.add_argument("--days", type=int, default=90, help="Days for cleanup action")
    parser.add_argument("--output", default="export.json", help="Output file for export")
    parser.add_argument("--table", default="tiktok_knowledge", help="Table name")
    
    args = parser.parse_args()
    
    db_config = {
        "host": "localhost",
        "port": "5432", 
        "database": "tiktok_learning",
        "user": "postgres",
        "password": ""  # Set your password
    }
    
    manager = TikTokKnowledgeManager(db_config)
    
    async def run_action():
        if args.action == "update":
            await manager.update_from_sources(TIKTOK_KNOWLEDGE_SOURCES, args.table)
        elif args.action == "stats":
            stats = await manager.get_knowledge_stats(args.table)
            print(json.dumps(stats, indent=2))
        elif args.action == "cleanup":
            await manager.cleanup_old_content(args.table, args.days)
        elif args.action == "export":
            await manager.export_knowledge_base(args.table, args.output)
        elif args.action == "find":
            if not args.query:
                print("Query required for find action")
                return
            results = await manager.find_similar_content(args.query, args.table)
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} (Similarity: {result['similarity']:.3f}) ---")
                print(f"Topics: {result['topics']}")
                print(f"Source: {result['source_url']}")
                print(f"Content: {result['content'][:200]}...")
    
    asyncio.run(run_action())
