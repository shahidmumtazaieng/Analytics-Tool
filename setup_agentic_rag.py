#!/usr/bin/env python3
"""
Setup script for TikTok Shop Agentic RAG System
Initializes databases, creates tables, and verifies configuration
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any
import asyncpg
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from agentic_rag_system.agent.db_utils import initialize_database, test_connection
from agentic_rag_system.agent.graph_utils import initialize_graph, test_graph_connection
from agentic_rag_system.agent.providers import validate_configuration, get_model_info

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_environment():
    """Check if all required environment variables are set."""
    logger.info("🔍 Checking environment configuration...")
    
    required_vars = [
        'GEMINI_API_KEY',
        'DATABASE_URL',
        'NEO4J_URI',
        'NEO4J_USER', 
        'NEO4J_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.info("Please check your .env file and ensure all required variables are set.")
        return False
    
    logger.info("✅ Environment configuration looks good!")
    return True


async def setup_postgresql():
    """Set up PostgreSQL database with pgvector extension."""
    logger.info("🐘 Setting up PostgreSQL database...")
    
    try:
        # Test basic connection
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            logger.error("❌ DATABASE_URL not found in environment")
            return False
        
        # Connect and check if database exists
        conn = await asyncpg.connect(database_url)
        
        # Check if pgvector extension is available
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
        )
        
        if not result:
            logger.warning("⚠️  pgvector extension not available. Please install it first.")
            logger.info("Run: CREATE EXTENSION vector; in your PostgreSQL database")
        
        await conn.close()
        
        # Initialize our database utilities
        await initialize_database()
        
        # Test connection
        if await test_connection():
            logger.info("✅ PostgreSQL setup completed successfully!")
            return True
        else:
            logger.error("❌ PostgreSQL connection test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ PostgreSQL setup failed: {e}")
        return False


async def setup_neo4j():
    """Set up Neo4j knowledge graph database."""
    logger.info("🕸️  Setting up Neo4j knowledge graph...")
    
    try:
        # Initialize graph connection
        await initialize_graph()
        
        # Test connection
        if await test_graph_connection():
            logger.info("✅ Neo4j setup completed successfully!")
            return True
        else:
            logger.warning("⚠️  Neo4j connection test failed - knowledge graph features may be limited")
            return True  # Don't fail setup if graph is unavailable
            
    except Exception as e:
        logger.warning(f"⚠️  Neo4j setup failed: {e} - continuing without knowledge graph")
        return True  # Don't fail setup if graph is unavailable


async def verify_ai_providers():
    """Verify AI provider configuration."""
    logger.info("🤖 Verifying AI provider configuration...")
    
    try:
        # Validate configuration
        validate_configuration()
        
        # Get model info
        model_info = get_model_info()
        logger.info(f"📊 Model configuration:")
        logger.info(f"   LLM Provider: {model_info['llm_provider']}")
        logger.info(f"   LLM Model: {model_info['llm_model']}")
        logger.info(f"   Embedding Provider: {model_info['embedding_provider']}")
        logger.info(f"   Embedding Model: {model_info['embedding_model']}")
        
        logger.info("✅ AI provider configuration verified!")
        return True
        
    except Exception as e:
        logger.error(f"❌ AI provider verification failed: {e}")
        return False


async def create_sample_data():
    """Create sample data for testing."""
    logger.info("📝 Creating sample data...")
    
    try:
        # Import the TikTok service
        from tiktok_rag_integration import TikTokLearningService
        
        service = TikTokLearningService()
        await service.initialize()
        
        # Sample YouTube transcript data
        sample_transcripts = [
            {
                "video_id": "sample_001",
                "title": "TikTok Shop Product Hunting Strategies 2024",
                "transcript": "In this video, I'll share the top 5 strategies for finding winning products on TikTok Shop. First, analyze trending hashtags and viral content. Second, use TikTok's Creative Center to identify popular products. Third, monitor competitor stores and their best-selling items. Fourth, leverage seasonal trends and holidays. Fifth, test products with small budgets before scaling.",
                "channel": "TikTok Shop Mastery",
                "duration": 600,
                "views": 50000,
                "upload_date": "2024-01-15",
                "tags": ["tiktok shop", "product hunting", "e-commerce", "dropshipping"],
                "url": "https://youtube.com/watch?v=sample_001"
            },
            {
                "video_id": "sample_002", 
                "title": "TikTok Shop Compliance Guide - Avoid Account Bans",
                "transcript": "TikTok Shop compliance is crucial for long-term success. Here are the key policies to follow: 1) Only sell authentic products with proper documentation. 2) Ensure all product descriptions are accurate and not misleading. 3) Follow TikTok's community guidelines in your content. 4) Respond to customer inquiries promptly. 5) Handle returns and refunds according to TikTok's policies. Violating these can result in account suspension or permanent bans.",
                "channel": "E-commerce Legal",
                "duration": 480,
                "views": 25000,
                "upload_date": "2024-01-20",
                "tags": ["tiktok shop", "compliance", "policies", "account safety"],
                "url": "https://youtube.com/watch?v=sample_002"
            }
        ]
        
        # Sample Facebook group data
        sample_facebook_data = [
            {
                "id": "fb_post_001",
                "type": "post",
                "content": "Just got my TikTok Shop account reinstated after 2 weeks! Here's what worked: 1) Submitted a detailed appeal explaining the misunderstanding. 2) Provided proof of product authenticity. 3) Showed evidence of good customer service. 4) Waited patiently and followed up professionally. Don't give up if you get banned - appeals do work!",
                "author": "Sarah_TikTokSeller",
                "group_name": "TikTok Shop Sellers United",
                "likes": 127,
                "comments_count": 23,
                "shares": 15,
                "post_date": "2024-01-25",
                "engagement_score": 0.85,
                "url": "https://facebook.com/groups/tiktokshop/posts/001"
            },
            {
                "id": "fb_comment_001",
                "type": "comment", 
                "content": "I've been selling on TikTok Shop for 6 months now. My best advice for new sellers: Start with trending products but make sure you can fulfill orders quickly. TikTok's algorithm favors shops with fast shipping and good reviews. Also, create authentic content - don't just repost other people's videos.",
                "author": "Mike_EcomExpert",
                "group_name": "TikTok Shop Success Stories",
                "likes": 45,
                "comments_count": 8,
                "shares": 3,
                "post_date": "2024-01-28",
                "engagement_score": 0.72,
                "url": "https://facebook.com/groups/tiktokshop/posts/002/comments/001"
            }
        ]
        
        # Ingest sample data
        youtube_result = await service.ingest_youtube_data(sample_transcripts)
        facebook_result = await service.ingest_facebook_data(sample_facebook_data)
        
        logger.info(f"✅ Sample data created successfully!")
        logger.info(f"   YouTube transcripts: {youtube_result.get('processed_count', 0)}")
        logger.info(f"   Facebook items: {facebook_result.get('processed_count', 0)}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️  Sample data creation failed: {e} - continuing without sample data")
        return True  # Don't fail setup if sample data fails


async def run_setup():
    """Run the complete setup process."""
    logger.info("🚀 Starting TikTok Shop Agentic RAG System setup...")
    
    steps = [
        ("Environment Check", check_environment),
        ("PostgreSQL Setup", setup_postgresql),
        ("Neo4j Setup", setup_neo4j),
        ("AI Providers Verification", verify_ai_providers),
        ("Sample Data Creation", create_sample_data)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"Step: {step_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await step_func()
            results[step_name] = result
            
            if not result and step_name in ["Environment Check", "PostgreSQL Setup", "AI Providers Verification"]:
                logger.error(f"❌ Critical step '{step_name}' failed. Stopping setup.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Step '{step_name}' failed with error: {e}")
            results[step_name] = False
            
            if step_name in ["Environment Check", "PostgreSQL Setup", "AI Providers Verification"]:
                logger.error("❌ Critical step failed. Stopping setup.")
                return False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Setup Summary")
    logger.info(f"{'='*50}")
    
    for step_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{step_name}: {status}")
    
    success_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    if success_count == total_count:
        logger.info(f"\n🎉 Setup completed successfully! ({success_count}/{total_count} steps passed)")
        logger.info("\n📋 Next steps:")
        logger.info("1. Start the backend server: uvicorn app:app --reload")
        logger.info("2. Test the TikTok learning endpoint: POST /chat/tiktok_learning")
        logger.info("3. Ingest your own YouTube transcripts and Facebook data")
        logger.info("4. Explore the agentic RAG capabilities!")
        return True
    else:
        logger.warning(f"\n⚠️  Setup completed with warnings. ({success_count}/{total_count} steps passed)")
        logger.info("Check the logs above for any issues that need attention.")
        return True


if __name__ == "__main__":
    try:
        success = asyncio.run(run_setup())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Setup failed with unexpected error: {e}")
        sys.exit(1)
