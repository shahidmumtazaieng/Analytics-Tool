<xaiArtifact>
<artifact_id>tiktok-rag-setup-script</artifact_id>
<title>TikTok RAG System Setup Script</title>
<contentType>text/python</contentType>

#!/usr/bin/env python3
"""
TikTok RAG System Setup Script
Initializes Pinecone vector database and loads sample TikTok knowledge data
"""

import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from tiktok_rag_integration import TikTokRAGSystem, TikTokKnowledgeItem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample TikTok knowledge data
SAMPLE_TIKTOK_KNOWLEDGE = [
    TikTokKnowledgeItem(
        id="tiktok_trends_2024_001",
        title="TikTok E-commerce Trends 2024",
        content="TikTok Shop integration is revolutionizing e-commerce. Key trends include live shopping events, influencer collaborations, and AI-powered product recommendations. Brands are seeing 3x higher engagement rates compared to traditional social media platforms. The platform's algorithm prioritizes authentic content and user engagement, making it ideal for product discovery and viral marketing campaigns.",
        category="trends",
        tags=["e-commerce", "tiktok-shop", "live-shopping", "influencer-marketing", "viral-marketing"],
        source="TikTok Business Blog",
        created_at=datetime.now()
    ),
    TikTokKnowledgeItem(
        id="viral_content_strategy_001",
        title="Creating Viral TikTok Content for Products",
        content="To create viral TikTok content for products: 1) Use trending sounds and hashtags to increase discoverability, 2) Show product benefits in the first 3 seconds to capture attention, 3) Include user-generated content to build trust, 4) Leverage TikTok's algorithm with consistent posting (2-3 times daily), 5) Engage with comments immediately to boost engagement rates, 6) Use vertical video format optimized for mobile viewing, 7) Incorporate storytelling elements that resonate with your target audience.",
        category="strategy",
        tags=["viral-content", "product-marketing", "tiktok-algorithm", "engagement", "content-strategy"],
        source="TikTok Marketing Guide",
        created_at=datetime.now()
    ),
    TikTokKnowledgeItem(
        id="tiktok_ads_best_practices_001",
        title="TikTok Advertising Best Practices",
        content="TikTok advertising best practices: Target Gen Z and Millennials (ages 16-34), use vertical video format (9:16 aspect ratio), keep ads under 15 seconds for maximum engagement, incorporate trending music and sounds, use native TikTok features like duets and challenges, focus on authentic storytelling over hard selling, leverage user-generated content in ads, and use precise targeting based on interests and behaviors. Successful campaigns often achieve 2-3x higher engagement than traditional social media ads.",
        category="advertising",
        tags=["tiktok-ads", "targeting", "video-format", "authenticity", "campaign-optimization"],
        source="TikTok Ads Platform",
        created_at=datetime.now()
    ),
    TikTokKnowledgeItem(
        id="tiktok_shop_optimization_001",
        title="TikTok Shop Optimization Strategies",
        content="TikTok Shop optimization strategies: 1) Optimize product listings with high-quality images and compelling descriptions, 2) Use trending hashtags and sounds in product videos, 3) Leverage influencer partnerships for product promotion, 4) Implement live shopping events to drive real-time sales, 5) Use TikTok's built-in analytics to track performance, 6) Create product bundles and limited-time offers, 7) Engage with customer reviews and feedback, 8) Cross-promote products across multiple TikTok accounts and platforms.",
        category="e-commerce",
        tags=["tiktok-shop", "product-optimization", "live-shopping", "influencer-partnerships", "analytics"],
        source="TikTok Shop Guide",
        created_at=datetime.now()
    ),
    TikTokKnowledgeItem(
        id="influencer_marketing_tiktok_001",
        title="Influencer Marketing on TikTok",
        content="Influencer marketing on TikTok: Partner with micro-influencers (10K-100K followers) for higher engagement rates, focus on authentic content creators who align with your brand values, use affiliate marketing programs to track ROI, create branded hashtag challenges to increase reach, leverage TikTok's Creator Marketplace for verified partnerships, provide creators with creative freedom while maintaining brand guidelines, and measure success through engagement rates, click-through rates, and conversion tracking.",
        category="influencer-marketing",
        tags=["influencer-marketing", "micro-influencers", "affiliate-marketing", "branded-challenges", "creator-marketplace"],
        source="TikTok Creator Hub",
        created_at=datetime.now()
    ),
    TikTokKnowledgeItem(
        id="tiktok_algorithm_insights_001",
        title="Understanding TikTok Algorithm for E-commerce",
        content="TikTok algorithm insights for e-commerce: The algorithm prioritizes content based on user interactions (likes, comments, shares, follows), video completion rates, and user preferences. To optimize for the algorithm: post consistently (2-3 times daily), use trending sounds and hashtags, engage with your audience through comments and duets, create content that encourages user interaction, optimize video length (15-60 seconds for maximum engagement), and focus on niche communities relevant to your products. The algorithm also favors content that keeps users on the platform longer.",
        category="algorithm",
        tags=["tiktok-algorithm", "content-optimization", "engagement", "user-interaction", "viral-potential"],
        source="TikTok Algorithm Research",
        created_at=datetime.now()
    ),
    TikTokKnowledgeItem(
        id="product_research_tiktok_001",
        title="Product Research Using TikTok Trends",
        content="Product research using TikTok trends: Monitor trending hashtags and sounds to identify emerging product categories, analyze viral product videos to understand consumer preferences, use TikTok's search function to discover trending products in your niche, track competitor content to identify successful product strategies, leverage TikTok's analytics to understand audience demographics and interests, and use trending challenges to test product-market fit before full-scale launches.",
        category="product-research",
        tags=["product-research", "trend-analysis", "market-research", "consumer-insights", "competitive-analysis"],
        source="TikTok Research Guide",
        created_at=datetime.now()
    ),
    TikTokKnowledgeItem(
        id="cross_platform_strategy_001",
        title="Cross-Platform E-commerce Strategy with TikTok",
        content="Cross-platform e-commerce strategy with TikTok: Use TikTok as a discovery platform to drive traffic to your main e-commerce site, integrate TikTok Shop with your existing online store, create platform-specific content while maintaining brand consistency, use TikTok's unique features (duets, challenges) to differentiate from other platforms, leverage TikTok's younger demographic to expand your customer base, and use TikTok analytics to inform content strategy across all platforms.",
        category="strategy",
        tags=["cross-platform", "e-commerce-strategy", "brand-consistency", "customer-acquisition", "platform-integration"],
        source="E-commerce Strategy Guide",
        created_at=datetime.now()
    ),
    TikTokKnowledgeItem(
        id="tiktok_analytics_guide_001",
        title="TikTok Analytics for E-commerce Success",
        content="TikTok analytics for e-commerce success: Track key metrics including video views, engagement rates, follower growth, click-through rates, and conversion rates. Use TikTok's built-in analytics to understand audience demographics, peak posting times, and content performance. Monitor trending hashtags and sounds to optimize content strategy. Analyze competitor performance to identify opportunities and best practices. Use third-party tools for deeper insights and cross-platform comparison.",
        category="analytics",
        tags=["tiktok-analytics", "performance-tracking", "audience-insights", "content-optimization", "competitive-analysis"],
        source="TikTok Analytics Guide",
        created_at=datetime.now()
    ),
    TikTokKnowledgeItem(
        id="tiktok_seo_optimization_001",
        title="TikTok SEO and Discovery Optimization",
        content="TikTok SEO and discovery optimization: Use relevant hashtags (mix of popular and niche), optimize video descriptions with keywords, create compelling thumbnails, use trending sounds and music, post consistently to maintain algorithm favor, engage with trending topics and challenges, use location tags for local businesses, create content that encourages user interaction, and optimize your TikTok profile with relevant keywords and bio information.",
        category="seo",
        tags=["tiktok-seo", "discovery-optimization", "hashtag-strategy", "keyword-optimization", "profile-optimization"],
        source="TikTok SEO Guide",
        created_at=datetime.now()
    )
]

async def setup_tiktok_rag_system():
    """Initialize TikTok RAG system with sample data"""
    
    # Check required environment variables
    required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these variables in your .env file")
        return False
    
    try:
        # Initialize RAG system
        logger.info("Initializing TikTok RAG System...")
        rag_system = TikTokRAGSystem(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "tiktok-learning"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_path=os.getenv("FINE_TUNED_MODEL_PATH", "./tiktok-learning-model")
        )
        
        # Add sample knowledge to Pinecone
        logger.info("Adding sample TikTok knowledge to vector database...")
        await rag_system.add_knowledge(SAMPLE_TIKTOK_KNOWLEDGE)
        
        # Test the system
        logger.info("Testing RAG system with sample query...")
        test_query = "What are the latest TikTok trends for e-commerce?"
        result = await rag_system.process_chat_message(test_query, "test_user")
        
        logger.info("✅ TikTok RAG System setup completed successfully!")
        logger.info(f"Test response: {result['response'][:200]}...")
        logger.info(f"Sources: {result['sources']}")
        logger.info(f"Confidence: {result['confidence']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup TikTok RAG System: {e}")
        return False

async def test_rag_functionality():
    """Test RAG functionality with various queries"""
    
    logger.info("Testing RAG functionality...")
    
    try:
        rag_system = TikTokRAGSystem(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "tiktok-learning"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        test_queries = [
            "How to create viral TikTok content for products?",
            "What are the best practices for TikTok advertising?",
            "How to optimize TikTok Shop for better sales?",
            "What are the latest TikTok trends for e-commerce?",
            "How to use TikTok for product research?"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: {query}")
            result = await rag_system.process_chat_message(query, "test_user")
            logger.info(f"Response: {result['response'][:150]}...")
            logger.info(f"Sources: {result['sources']}")
            logger.info(f"Confidence: {result['confidence']}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")

def create_env_template():
    """Create environment variables template"""
    
    env_template = """# TikTok RAG System Environment Variables

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=tiktok-learning
PINECONE_ENVIRONMENT=us-west1-gcp

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Model Configuration
FINE_TUNED_MODEL_PATH=./tiktok-learning-model

# Optional: Model Training
WANDB_API_KEY=your-wandb-api-key
WANDB_PROJECT=tiktok-learning-rag
"""
    
    with open("env.tiktok.rag.template", "w") as f:
        f.write(env_template)
    
    logger.info("Created env.tiktok.rag.template file")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TikTok RAG System Setup")
    parser.add_argument("--setup", action="store_true", help="Setup RAG system with sample data")
    parser.add_argument("--test", action="store_true", help="Test RAG functionality")
    parser.add_argument("--create-env", action="store_true", help="Create environment template")
    
    args = parser.parse_args()
    
    if args.create_env:
        create_env_template()
    
    if args.setup:
        success = asyncio.run(setup_tiktok_rag_system())
        if success:
            logger.info("Setup completed successfully!")
        else:
            logger.error("Setup failed!")
    
    if args.test:
        asyncio.run(test_rag_functionality())
    
    if not any([args.setup, args.test, args.create_env]):
        # Default: run setup
        success = asyncio.run(setup_tiktok_rag_system())
        if success:
            logger.info("Setup completed successfully!")
        else:
            logger.error("Setup failed!") 