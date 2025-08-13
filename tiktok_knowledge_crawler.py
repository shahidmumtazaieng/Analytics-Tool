"""
tiktok_knowledge_crawler.py
---------------------------
Enhanced TikTok Shop knowledge crawler using Crawl4AI for building comprehensive knowledge base.
Specifically designed for TikTok Shop, e-commerce, and social commerce content.

Features:
- TikTok Shop specific content detection and extraction
- Enhanced chunking for e-commerce content
- Metadata enrichment for better RAG performance
- Integration with existing PostgreSQL + pgvector setup
- Content quality filtering for TikTok-related information

Usage:
    python tiktok_knowledge_crawler.py <URL> [--collection ...] [--focus tiktok_shop]
"""

import argparse
import sys
import re
import asyncio
import os
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from datetime import datetime
import json

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Database imports
import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

class TikTokKnowledgeCrawler:
    """Enhanced crawler specifically for TikTok Shop and e-commerce content"""
    
    def __init__(self, db_config: Dict[str, str], embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_config = db_config
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tiktok_keywords = [
            "tiktok shop", "tiktok seller", "tiktok commerce", "tiktok business",
            "social commerce", "live shopping", "creator economy", "influencer marketing",
            "product hunting", "dropshipping", "tiktok ads", "tiktok marketing",
            "tiktok trends", "viral products", "tiktok algorithm", "content creation",
            "tiktok policies", "seller center", "tiktok analytics", "conversion rate"
        ]
    
    def is_tiktok_relevant(self, content: str) -> bool:
        """Check if content is relevant to TikTok Shop/e-commerce"""
        content_lower = content.lower()
        relevance_score = sum(1 for keyword in self.tiktok_keywords if keyword in content_lower)
        return relevance_score >= 2  # At least 2 TikTok-related keywords
    
    def smart_chunk_tiktok_content(self, markdown: str, max_len: int = 1000) -> List[Dict[str, Any]]:
        """Enhanced chunking specifically for TikTok/e-commerce content"""
        
        def split_by_header(md, header_pattern):
            indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
            indices.append(len(md))
            return [md[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1) if md[indices[i]:indices[i+1]].strip()]
        
        chunks = []
        
        # First, try to split by headers
        for h1 in split_by_header(markdown, r'^# .+$'):
            if len(h1) > max_len:
                for h2 in split_by_header(h1, r'^## .+$'):
                    if len(h2) > max_len:
                        for h3 in split_by_header(h2, r'^### .+$'):
                            if len(h3) > max_len:
                                # Split by sentences for better context preservation
                                sentences = re.split(r'(?<=[.!?])\s+', h3)
                                current_chunk = ""
                                for sentence in sentences:
                                    if len(current_chunk + sentence) <= max_len:
                                        current_chunk += sentence + " "
                                    else:
                                        if current_chunk.strip():
                                            chunks.append(current_chunk.strip())
                                        current_chunk = sentence + " "
                                if current_chunk.strip():
                                    chunks.append(current_chunk.strip())
                            else:
                                chunks.append(h3)
                    else:
                        chunks.append(h2)
            else:
                chunks.append(h1)
        
        # Process chunks and add metadata
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            # Extract metadata
            headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
            header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''
            
            # Calculate relevance score
            relevance_score = sum(1 for keyword in self.tiktok_keywords if keyword.lower() in chunk.lower())
            
            # Extract key topics
            topics = self.extract_topics(chunk)
            
            chunk_data = {
                "content": chunk,
                "headers": header_str,
                "char_count": len(chunk),
                "word_count": len(chunk.split()),
                "relevance_score": relevance_score,
                "is_tiktok_relevant": self.is_tiktok_relevant(chunk),
                "topics": topics,
                "chunk_index": i
            }
            
            processed_chunks.append(chunk_data)
        
        return processed_chunks
    
    def extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        topics = []
        content_lower = content.lower()
        
        # TikTok Shop specific topics
        topic_patterns = {
            "product_hunting": ["product hunting", "winning products", "product research"],
            "compliance": ["policy", "violation", "suspended", "banned", "compliance"],
            "marketing": ["marketing", "advertising", "promotion", "campaign"],
            "analytics": ["analytics", "metrics", "performance", "conversion"],
            "trends": ["trending", "viral", "popular", "trend"],
            "seller_tools": ["seller center", "dashboard", "tools", "features"],
            "monetization": ["monetization", "revenue", "profit", "earnings"],
            "content_creation": ["content", "video", "creative", "production"]
        }
        
        for topic, keywords in topic_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    async def crawl_with_tiktok_focus(self, url: str, focus_type: str = "tiktok_shop") -> List[Dict[str, Any]]:
        """Crawl with TikTok Shop focus using Crawl4AI"""
        
        # Configure browser for TikTok content
        browser_config = BrowserConfig(
            headless=True,
            verbose=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Configure crawler with content filtering
        if focus_type == "tiktok_shop":
            content_filter = BM25ContentFilter(
                user_query="TikTok Shop e-commerce social commerce seller business marketing",
                bm25_threshold=1.0
            )
        else:
            content_filter = PruningContentFilter(
                threshold=0.48,
                threshold_type="fixed",
                min_word_threshold=50
            )
        
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            markdown_generator=DefaultMarkdownGenerator(content_filter=content_filter),
            word_count_threshold=50,  # Minimum words per page
            only_text=False,  # Include images and media
            process_iframes=True,  # Process embedded content
            remove_overlay_elements=True  # Remove popups/overlays
        )
        
        results = []
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            try:
                result = await crawler.arun(url=url, config=run_config)
                
                if result.success and result.markdown:
                    # Use fit_markdown for better content quality
                    markdown_content = result.markdown.fit_markdown or result.markdown.raw_markdown
                    
                    # Only process if content is TikTok relevant
                    if self.is_tiktok_relevant(markdown_content):
                        results.append({
                            'url': url,
                            'markdown': markdown_content,
                            'title': result.metadata.get('title', ''),
                            'description': result.metadata.get('description', ''),
                            'keywords': result.metadata.get('keywords', ''),
                            'crawled_at': datetime.now().isoformat(),
                            'success': True
                        })
                    else:
                        print(f"Content not TikTok relevant: {url}")
                        results.append({
                            'url': url,
                            'markdown': None,
                            'error': 'Content not TikTok relevant',
                            'success': False
                        })
                else:
                    print(f"Failed to crawl {url}: {result.error_message}")
                    results.append({
                        'url': url,
                        'markdown': None,
                        'error': result.error_message,
                        'success': False
                    })
                    
            except Exception as e:
                print(f"Error crawling {url}: {str(e)}")
                results.append({
                    'url': url,
                    'markdown': None,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    async def crawl_sitemap_tiktok_focused(self, sitemap_url: str, max_urls: int = 100) -> List[Dict[str, Any]]:
        """Crawl sitemap with TikTok focus"""
        try:
            resp = requests.get(sitemap_url, timeout=30)
            urls = []
            
            if resp.status_code == 200:
                tree = ElementTree.fromstring(resp.content)
                all_urls = [loc.text for loc in tree.findall('.//{*}loc')]
                
                # Filter URLs that might contain TikTok content
                tiktok_url_patterns = [
                    'tiktok', 'social', 'commerce', 'shop', 'seller', 'marketing',
                    'ecommerce', 'business', 'guide', 'tutorial', 'strategy'
                ]
                
                for url in all_urls[:max_urls]:
                    url_lower = url.lower()
                    if any(pattern in url_lower for pattern in tiktok_url_patterns):
                        urls.append(url)
                
                print(f"Found {len(urls)} TikTok-relevant URLs from sitemap")
            
            # Crawl filtered URLs
            results = []
            for url in urls:
                url_results = await self.crawl_with_tiktok_focus(url)
                results.extend(url_results)
                
                # Add delay to be respectful
                await asyncio.sleep(1)
            
            return results
            
        except Exception as e:
            print(f"Error processing sitemap {sitemap_url}: {str(e)}")
            return []
    
    async def insert_into_postgres(self, chunks: List[Dict[str, Any]], table_name: str = "tiktok_knowledge"):
        """Insert chunks into PostgreSQL with pgvector"""
        try:
            conn = await asyncpg.connect(**self.db_config)
            
            # Create table if not exists
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB,
                    source_url TEXT,
                    headers TEXT,
                    topics TEXT[],
                    relevance_score INTEGER,
                    char_count INTEGER,
                    word_count INTEGER,
                    is_tiktok_relevant BOOLEAN,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create index for vector similarity search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
                ON {table_name} USING ivfflat (embedding vector_cosine_ops)
            """)
            
            # Insert chunks in batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Generate embeddings
                contents = [chunk['content'] for chunk in batch]
                embeddings = self.embedding_model.encode(contents)
                
                # Prepare data for insertion
                insert_data = []
                for j, chunk in enumerate(batch):
                    embedding_list = embeddings[j].tolist()
                    
                    insert_data.append((
                        chunk['content'],
                        embedding_list,
                        json.dumps({
                            'headers': chunk.get('headers', ''),
                            'topics': chunk.get('topics', []),
                            'chunk_index': chunk.get('chunk_index', 0),
                            'crawled_at': chunk.get('crawled_at', datetime.now().isoformat())
                        }),
                        chunk.get('source_url', ''),
                        chunk.get('headers', ''),
                        chunk.get('topics', []),
                        chunk.get('relevance_score', 0),
                        chunk.get('char_count', 0),
                        chunk.get('word_count', 0),
                        chunk.get('is_tiktok_relevant', False)
                    ))
                
                # Insert batch
                await conn.executemany(f"""
                    INSERT INTO {table_name} 
                    (content, embedding, metadata, source_url, headers, topics, 
                     relevance_score, char_count, word_count, is_tiktok_relevant)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, insert_data)
                
                print(f"Inserted batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            await conn.close()
            print(f"Successfully inserted {len(chunks)} chunks into {table_name}")
            
        except Exception as e:
            print(f"Error inserting into PostgreSQL: {str(e)}")
            raise

def is_sitemap(url: str) -> bool:
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    return url.endswith('.txt') or url.endswith('.md')

async def main():
    parser = argparse.ArgumentParser(description="TikTok Knowledge Crawler with Crawl4AI")
    parser.add_argument("url", help="URL to crawl (regular, .txt, or sitemap)")
    parser.add_argument("--table", default="tiktok_knowledge", help="PostgreSQL table name")
    parser.add_argument("--focus", default="tiktok_shop", choices=["tiktok_shop", "general"], help="Content focus")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max chunk size (chars)")
    parser.add_argument("--max-urls", type=int, default=100, help="Max URLs from sitemap")
    parser.add_argument("--db-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--db-port", default="5432", help="PostgreSQL port")
    parser.add_argument("--db-name", default="tiktok_learning", help="PostgreSQL database")
    parser.add_argument("--db-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--db-password", help="PostgreSQL password")
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        "host": args.db_host,
        "port": args.db_port,
        "database": args.db_name,
        "user": args.db_user,
        "password": args.db_password or os.getenv("POSTGRES_PASSWORD", "")
    }
    
    # Initialize crawler
    crawler = TikTokKnowledgeCrawler(db_config)
    
    # Determine crawl strategy
    url = args.url
    if is_sitemap(url):
        print(f"Crawling sitemap with TikTok focus: {url}")
        crawl_results = await crawler.crawl_sitemap_tiktok_focused(url, max_urls=args.max_urls)
    else:
        print(f"Crawling single URL with TikTok focus: {url}")
        crawl_results = await crawler.crawl_with_tiktok_focus(url, focus_type=args.focus)
    
    # Process successful results
    all_chunks = []
    for result in crawl_results:
        if result['success'] and result['markdown']:
            chunks = crawler.smart_chunk_tiktok_content(result['markdown'], max_len=args.chunk_size)
            
            # Add source URL to each chunk
            for chunk in chunks:
                chunk['source_url'] = result['url']
                chunk['crawled_at'] = result.get('crawled_at', datetime.now().isoformat())
            
            # Filter only TikTok relevant chunks
            relevant_chunks = [chunk for chunk in chunks if chunk['is_tiktok_relevant']]
            all_chunks.extend(relevant_chunks)
    
    if not all_chunks:
        print("No TikTok-relevant content found to insert.")
        return
    
    print(f"Found {len(all_chunks)} TikTok-relevant chunks to insert")
    
    # Insert into PostgreSQL
    await crawler.insert_into_postgres(all_chunks, table_name=args.table)
    
    print(f"Successfully processed and inserted TikTok knowledge from {url}")

if __name__ == "__main__":
    asyncio.run(main())
