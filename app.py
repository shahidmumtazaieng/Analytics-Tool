import os
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages, HumanMessage, AIMessage
from firebase_admin import credentials, initialize_app, auth, firestore, storage
import firebase_admin
import base64
from PIL import Image
import io
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json"))
    initialize_app(cred)
db = firestore.client()

# FastAPI app
app = FastAPI(
    title="E-commerce Product Research SaaS",
    description="AI-powered e-commerce product research platform with multi-platform analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Pydantic models
class User(BaseModel):
    email: str = Field(..., description="User email address")
    bright_data_api_key: str = Field(..., description="Bright Data API key")
    openai_api_key: str = Field(..., description="OpenAI API key")

class ChatRequest(BaseModel):
    tool: str = Field(..., description="Tool identifier")
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class ImageVerification(BaseModel):
    youtube_url: str = Field(..., description="YouTube channel URL")
    screenshot_data: str = Field(..., description="Base64 encoded screenshot")

class ResearchRequest(BaseModel):
    tool: str = Field(..., description="Tool identifier")
    filters: Optional[Dict[str, Any]] = Field(None, description="Research filters") 

# Dependency to verify user and get Firestore data
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        decoded = auth.verify_id_token(token)
        user_id = decoded["uid"]

        # Create or update user profile using new user management system
        user_profile = await user_manager.create_user_profile(decoded)

        # Check user access permissions
        access_check = await user_manager.check_user_access(user_id)

        # Add access information to user data
        user_data = {
            "uid": user_id,
            "email": decoded["email"],
            "profile": user_profile.__dict__,
            "access": access_check
        }

        return user_data

    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")


# Dependency for endpoints that require full access (authorized + API configured)
async def get_authorized_user(user_data: Dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency that requires user to be fully authorized and have APIs configured"""
    access = user_data.get("access", {})

    if not access.get("has_access"):
        reason = access.get("reason", "unknown")
        action = access.get("requires_action", "contact_support")

        if reason == "authorization_pending":
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Authorization required",
                    "message": "Please complete YouTube channel verification",
                    "action": "youtube_verification"
                }
            )
        elif reason == "api_configuration_needed":
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "API configuration required",
                    "message": "Please configure your API keys",
                    "action": "configure_apis"
                }
            )
        else:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Access denied",
                    "message": f"Account status: {reason}",
                    "action": action
                }
            )

    return user_data

# Initialize Bright Data client
async def get_bright_data_client(api_key: str) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "bright_data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": api_key,
                    "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                    "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser")
                },
                "transport": "stdio",
            }
        }
    )

# Tool configuration with enhanced prompts
TOOL_CONFIG = {
    "global-research": {
        "platform": "global",
        "tools": [
            "web_data_amazon_product_search", "web_data_walmart_product", 
            "web_data_ebay_product", "web_data_homedepot_products", 
            "web_data_zara_products", "web_data_etsy_products",
            "web_data_bestbuy_products", "web_data_tiktok_shop"
        ],
        "prompt": """You are a global e-commerce research analyst. Use all available tools to:
        - Aggregate best-selling products across platforms
        - Identify product gaps (products on one platform but not others)
        - Rank categories by sales, profitability, and competition
        - Analyze cross-platform trends and opportunities
        
        Return JSON with: 
        - product_list: [name, category, platform, price, sales, sellers, reviews, gap_status]
        - category_rankings: [name, total_sales, avg_profitability, avg_competition, top_products, gap_opportunities]
        - trends: [trend_name, description, platforms_affected, opportunity_score]
        - metadata: {sort_options: [profitability, demand, competition], filter_options: [platform, category, gap_status]}"""
    },
    "product-gap-finder": {
        "platform": "global",
        "tools": [
            "web_data_amazon_product_search", "web_data_walmart_product", 
            "web_data_ebay_product", "web_data_homedepot_products", 
            "web_data_zara_products", "web_data_etsy_products",
            "web_data_bestbuy_products", "web_data_tiktok_shop"
        ],
        "prompt": """You are a product gap analyst. Use all available tools to:
        - Identify products selling well on one platform but absent or underperforming on others
        - Calculate gap opportunity scores based on demand and competition
        - Provide detailed gap analysis with cross-platform comparisons
        
        Return JSON with:
        - product_list: [name, category, platform, price, sales, sellers, reviews, gap_status, gap_opportunity_score]
        - gap_analysis: [gap_type, source_platform, target_platforms, opportunity_score, estimated_profit]
        - recommendations: [action, platform, reasoning, expected_outcome]"""
    },
    "amazon-best-sellers": {
        "platform": "amazon",
        "tools": ["web_data_amazon_product_search"],
        "prompt": """You are an Amazon research analyst. Use web_data_amazon_product_search to:
        - Identify best-selling products (high sales volume, positive reviews)
        - Find low-competition products (<10 sellers, <500 reviews)
        - Identify high-profit products (3x markup after 30% fees, 20% costs)
        - Analyze Amazon-specific trends and opportunities
        
        Return JSON with:
        - product_list: [name, category, price, sales, sellers, reviews, profitability_score, competition_score, demand_score]
        - top_categories: [name, total_sales, avg_profitability, avg_competition, top_products]
        - amazon_trends: [trend_name, description, affected_categories, opportunity_score]"""
    },
    "amazon-product-gaps": {
        "platform": "amazon",
        "tools": [
            "web_data_amazon_product_search", "web_data_walmart_product", 
            "web_data_ebay_product", "web_data_homedepot_products", 
            "web_data_zara_products", "web_data_etsy_products",
            "web_data_bestbuy_products", "web_data_tiktok_shop"
        ],
        "prompt": """You are an Amazon gap analyst. Use web_data_amazon_product_search and other tools to:
        - Identify products selling well on Amazon but absent or underperforming on other platforms
        - Find products on other platforms that could succeed on Amazon
        - Calculate cross-platform opportunity scores
        
        Return JSON with:
        - product_list: [name, category, platform, price, sales, sellers, reviews, gap_status, amazon_opportunity_score]
        - amazon_gaps: [product_name, missing_on_amazon, source_platform, opportunity_score]
        - recommendations: [action, product, platform, reasoning]"""
    },
    "ebay-best-sellers": {
        "platform": "ebay",
        "tools": ["web_data_ebay_product"],
        "prompt": """You are an eBay research analyst. Use web_data_ebay_product to:
        - Identify best-selling products (high sales volume, positive reviews)
        - Find low-competition products (<10 sellers, <500 reviews)
        - Identify high-profit products (3x markup after 30% fees, 20% costs)
        - Analyze eBay-specific trends and auction dynamics
        
        Return JSON with:
        - product_list: [name, category, price, sales, sellers, reviews, profitability_score, competition_score, demand_score]
        - top_categories: [name, total_sales, avg_profitability, avg_competition, top_products]
        - ebay_trends: [trend_name, description, affected_categories, opportunity_score]"""
    },
    "walmart-best-sellers": {
        "platform": "walmart",
        "tools": ["web_data_walmart_product"],
        "prompt": """You are a Walmart research analyst. Use web_data_walmart_product to:
        - Identify best-selling products (high sales volume, positive reviews)
        - Find low-competition products (<10 sellers, <500 reviews)
        - Identify high-profit products (3x markup after 30% fees, 20% costs)
        - Analyze Walmart-specific trends and retail dynamics
        
        Return JSON with:
        - product_list: [name, category, price, sales, sellers, reviews, profitability_score, competition_score, demand_score]
        - top_categories: [name, total_sales, avg_profitability, avg_competition, top_products]
        - walmart_trends: [trend_name, description, affected_categories, opportunity_score]"""
    },
    "homedepot-best-sellers": {
        "platform": "homedepot",
        "tools": ["web_data_homedepot_products"],
        "prompt": """You are a Home Depot research analyst. Use web_data_homedepot_products to:
        - Identify best-selling products (high sales volume, positive reviews)
        - Find low-competition products (<10 sellers, <500 reviews)
        - Identify high-profit products (3x markup after 30% fees, 20% costs)
        - Analyze Home Depot-specific trends and DIY market dynamics
        
        Return JSON with:
        - product_list: [name, category, price, sales, sellers, reviews, profitability_score, competition_score, demand_score]
        - top_categories: [name, total_sales, avg_profitability, avg_competition, top_products]
        - homedepot_trends: [trend_name, description, affected_categories, opportunity_score]"""
    },
    "zara-best-sellers": {
        "platform": "zara",
        "tools": ["web_data_zara_products"],
        "prompt": """You are a Zara research analyst. Use web_data_zara_products to:
        - Identify best-selling products (high sales volume, positive reviews)
        - Find low-competition products (<10 sellers, <500 reviews)
        - Identify high-profit products (3x markup after 30% fees, 20% costs)
        - Analyze Zara-specific trends and fashion market dynamics
        
        Return JSON with:
        - product_list: [name, category, price, sales, sellers, reviews, profitability_score, competition_score, demand_score]
        - top_categories: [name, total_sales, avg_profitability, avg_competition, top_products]
        - zara_trends: [trend_name, description, affected_categories, opportunity_score]"""
    },
    "etsy-best-sellers": {
        "platform": "etsy",
        "tools": ["web_data_etsy_products"],
        "prompt": """You are an Etsy research analyst. Use web_data_etsy_products to:
        - Identify best-selling products (high sales volume, positive reviews)
        - Find low-competition products (<10 sellers, <500 reviews)
        - Identify high-profit products (3x markup after 30% fees, 20% costs)
        - Analyze Etsy-specific trends and handmade market dynamics
        
        Return JSON with:
        - product_list: [name, category, price, sales, sellers, reviews, profitability_score, competition_score, demand_score]
        - top_categories: [name, total_sales, avg_profitability, avg_competition, top_products]
        - etsy_trends: [trend_name, description, affected_categories, opportunity_score]"""
    },
    "bestbuy-best-sellers": {
        "platform": "bestbuy",
        "tools": ["web_data_bestbuy_products"],
        "prompt": """You are a Best Buy research analyst. Use web_data_bestbuy_products to:
        - Identify best-selling products (high sales volume, positive reviews)
        - Find low-competition products (<10 sellers, <500 reviews)
        - Identify high-profit products (3x markup after 30% fees, 20% costs)
        - Analyze Best Buy-specific trends and electronics market dynamics
        
        Return JSON with:
        - product_list: [name, category, price, sales, sellers, reviews, profitability_score, competition_score, demand_score]
        - top_categories: [name, total_sales, avg_profitability, avg_competition, top_products]
        - bestbuy_trends: [trend_name, description, affected_categories, opportunity_score]"""
    },
    "tiktok-best-sellers": {
        "platform": "tiktok",
        "tools": ["web_data_tiktok_shop"],
        "prompt": """You are a TikTok Shop research analyst. Use web_data_tiktok_shop to:
        - Identify best-selling products (high sales volume, positive reviews)
        - Find low-competition products (<10 sellers, <500 reviews)
        - Identify high-profit products (3x markup after 30% fees, 20% costs)
        - Analyze TikTok-specific trends and viral product dynamics
        
        Return JSON with:
        - product_list: [name, category, price, sales, sellers, reviews, profitability_score, competition_score, demand_score]
        - top_categories: [name, total_sales, avg_profitability, avg_competition, top_products]
        - tiktok_trends: [trend_name, description, affected_categories, opportunity_score]"""
    },
    "tiktok-learning": {
        "platform": "tiktok",
        "tools": ["web_data_tiktok_shop"],
        "prompt": """You are a TikTok learning chatbot with RAG, fine-tuned on TikTok Shop data and e-commerce trends. 
        Use web_data_tiktok_shop and RAG context to answer questions about:
        - Trending products and viral marketing strategies
        - TikTok Shop optimization techniques
        - E-commerce trends and consumer behavior
        - Product research and validation methods
        - Cross-platform selling strategies
        
        Return JSON with:
        - answer: Detailed response with actionable insights
        - sources: [source_name, relevance_score, key_insights]
        - recommendations: [action, reasoning, expected_outcome]
        - related_topics: [topic_name, description, relevance_score]"""
    }
} 

# Image recognition for YouTube subscription verification
async def verify_youtube_subscription(youtube_url: str, screenshot_data: str) -> bool:
    try:
        logger.info(f"Verifying YouTube subscription for URL: {youtube_url}")
        
        # Decode base64 image
        image_data = base64.b64decode(screenshot_data.split(',')[1] if ',' in screenshot_data else screenshot_data)
        image = Image.open(io.BytesIO(image_data))
        
        # TODO: Replace with Google Cloud Vision API
        # For now, implement basic image analysis
        # This is a placeholder - replace with actual Google Cloud Vision implementation
        
        # Mock verification logic
        # In production, use Google Cloud Vision to:
        # 1. Extract text from image
        # 2. Verify channel name matches expected channel
        # 3. Check for "Subscribed" status
        
        # Placeholder verification
        return True
        
    except Exception as e:
        logger.error(f"Image verification error: {str(e)}")
        return False

# Pretty print messages for LangGraph
def pretty_print_messages(update, last_message=False):
    messages = []
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return messages
        is_subgraph = True
    for node_name, node_update in update.items():
        node_messages = convert_to_messages(node_update["messages"])
        if last_message:
            node_messages = node_messages[-1:]
        for m in node_messages:
            messages.append({"node": node_name, "content": m.content})
    return messages

# LangGraph agent system for a specific tool
async def run_tool_agent(tool: str, query: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
    config = TOOL_CONFIG.get(tool)
    if not config:
        raise HTTPException(status_code=400, detail="Invalid tool")
    
    try:
        client = await get_bright_data_client(user_data["bright_data_api_key"])
        tools = await client.get_tools()
        model = init_chat_model(model="openai:gpt-4", api_key=user_data["openai_api_key"])

        # Tool-specific research agent
        research_agent = create_react_agent(
            model,
            tools,
            prompt=config["prompt"],
            name=f"{tool}_research_agent"
        )

        # Analytics agent
        analytics_agent = create_react_agent(
            model,
            tools,
            prompt="""You are an e-commerce analytics expert. Analyze:
            - Profitability: Calculate margins (3x markup after 30% fees, 20% costs)
            - Competition: Score (<10 sellers, <500 reviews = low competition)
            - Demand: Score based on sales volume and trends
            - Gaps: Confirm products missing on specific platforms (if applicable)
            
            Return JSON with: 
            - product_list: [name, category, platform, price, profitability_score, competition_score, demand_score, gap_status]
            - analytics_summary: {total_products, avg_profitability, avg_competition, avg_demand, top_opportunities}
            - recommendations: [action, reasoning, expected_outcome]""",
            name="analytics_agent"
        )

        # Chat agent
        chat_agent = create_react_agent(
            model,
            tools,
            prompt=f"""You are a chat assistant for the {tool.replace('-', ' ').title()} tool. 
            Answer queries about products, categories, trends, or gaps using the latest data. 
            Provide concise, actionable responses in JSON or plain text as requested.
            
            Return JSON with:
            - answer: Detailed response
            - data_points: [relevant_data_point, source, confidence]
            - suggestions: [next_query, reasoning]""",
            name=f"{tool}_chat_agent"
        )

        # TikTok learning chatbot with Enhanced Agentic RAG
        tiktok_learning_agent = create_react_agent(
            model,
            tools,
            prompt="""You are an advanced TikTok Shop learning assistant with Agentic RAG capabilities.
            You have access to comprehensive knowledge from YouTube video transcripts and Facebook group discussions.

            Your expertise includes:
            - Product hunting and trend analysis
            - TikTok Shop compliance and policy guidance
            - Account reinstatement strategies
            - Marketing and growth strategies
            - Viral content creation techniques

            Use the agentic RAG system to provide detailed, actionable insights backed by real experiences
            from successful TikTok Shop sellers and comprehensive knowledge sources.

            Return JSON with:
            - answer: Comprehensive response with step-by-step guidance
            - sources: [source_name, source_type, relevance_score, key_insights]
            - tools_used: [tool_name, purpose, results_summary]
            - recommendations: [action, reasoning, expected_outcome, timeline]
            - related_topics: [topic_name, description, relevance_score]
            - confidence_score: Overall confidence in the response (0-1)""",
            name="tiktok_agentic_learning_agent"
        )

        # UI formatter agent
        ui_formatter_agent = create_react_agent(
            model,
            tools,
            prompt="""You are a UI data formatter. Format data for a React frontend in JSON:
            - product_list: [name, category, platform, price, profitability_score, competition_score, demand_score, gap_status]
            - category_rankings: [name, total_sales, avg_profitability, avg_competition, top_products, gap_opportunities]
            - metadata: {sort_options: [profitability, demand, competition], filter_options: [platform, category, gap_status]}
            - ui_state: {current_sort, current_filters, pagination_info}""",
            name="ui_formatter_agent"
        )

        # Supervisor
        agents = [research_agent, analytics_agent, chat_agent, tiktok_learning_agent, ui_formatter_agent]
        supervisor = create_supervisor(
            model=init_chat_model("openai:gpt-4", api_key=user_data["openai_api_key"]),
            agents=agents,
            prompt=(
                f"You are a supervisor for the {tool.replace('-', ' ').title()} tool:\n"
                f"- {tool}_research_agent: Handles research for {tool}.\n"
                f"- analytics_agent: Analyzes profitability, competition, demand, gaps.\n"
                f"- {tool}_chat_agent: Answers queries for {tool}.\n"
                f"- tiktok_agentic_learning_agent: Advanced TikTok Shop learning with Agentic RAG, knowledge graphs, and multi-source insights.\n"
                f"- ui_formatter_agent: Formats data for UI.\n"
                f"Assign tasks based on query. For TikTok Shop learning, compliance, product hunting, or strategy questions, use the agentic learning agent. Do not call agents in parallel."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile()

        result = []
        for chunk in supervisor.stream({"messages": [{"role": "user", "content": query}]}):
            result.extend(pretty_print_messages(chunk, last_message=True))
        
        return result[-1]["content"] if result else {}
        
    except Exception as e:
        logger.error(f"Tool agent error for {tool}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed for {tool}")

# API Endpoints
@app.post("/login")
async def login(user: User):
    try:
        # Create user document with email as ID (sanitized)
        user_id = user.email.replace(".", "_").replace("@", "_")
        user_ref = db.collection("users").document(user_id)
        
        user_ref.set({
            "email": user.email,
            "bright_data_api_key": user.bright_data_api_key,
            "openai_api_key": user.openai_api_key,
            "active": False,
            "created_at": datetime.now(),
            "last_login": datetime.now()
        })
        
        return {"message": "User registered successfully", "user_id": user_id}
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=400, detail="Registration failed")

# Enhanced Authorization and API Configuration Endpoints

@app.get("/api/auth/status")
async def get_auth_status(user_data: Dict = Depends(get_current_user)):
    """Get user's authentication and authorization status"""
    try:
        access = user_data.get("access", {})
        profile = user_data.get("profile", {})

        return {
            "success": True,
            "user": {
                "uid": user_data["uid"],
                "email": user_data["email"],
                "status": profile.get("status"),
                "api_configured": profile.get("api_configured", False)
            },
            "access": access,
            "youtube_channel": user_manager.get_youtube_channel_info()
        }
    except Exception as e:
        logger.error(f"Auth status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get auth status")


@app.get("/api/auth/youtube-verification-info")
async def get_youtube_verification_info(user_data: Dict = Depends(get_current_user)):
    """Get YouTube verification instructions and channel info"""
    try:
        instructions = await youtube_verification_ai.get_verification_instructions()
        return {
            "success": True,
            "instructions": instructions
        }
    except Exception as e:
        logger.error(f"YouTube info error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get YouTube info")


@app.post("/api/auth/verify-youtube")
async def verify_youtube_subscription(
    screenshot_file: UploadFile = File(...),
    user_data: Dict = Depends(get_current_user)
):
    """Verify YouTube subscription using AI-powered screenshot analysis"""
    try:
        # Read screenshot data
        screenshot_data = await screenshot_file.read()

        # Verify subscription using AI
        result, details = await youtube_verification_ai.verify_subscription_screenshot(
            screenshot_data,
            user_data["uid"]
        )

        # Update user authorization status
        if result == AuthorizationResult.APPROVED:
            await user_manager.update_authorization_status(
                user_data["uid"],
                UserStatus.API_PENDING,  # Move to API configuration step
                details
            )

            return {
                "success": True,
                "result": "approved",
                "message": "YouTube subscription verified! Please configure your API keys.",
                "next_step": "api_configuration",
                "details": details
            }
        else:
            await user_manager.update_authorization_status(
                user_data["uid"],
                UserStatus.INACTIVE,
                details
            )

            return {
                "success": False,
                "result": result.value,
                "message": f"Verification failed: {result.value}",
                "details": details,
                "retry_allowed": True
            }

    except Exception as e:
        logger.error(f"YouTube verification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Verification failed")


@app.post("/api/auth/configure-apis")
async def configure_user_apis(
    request: Dict[str, Any],
    user_data: Dict = Depends(get_current_user)
):
    """Configure user's API keys"""
    try:
        # Extract API keys from request
        api_config = APIConfiguration(
            anthropic_api_key=request.get("anthropic_api_key"),
            bright_data_api_key=request.get("bright_data_api_key"),
            openai_api_key=request.get("openai_api_key"),
            gemini_api_key=request.get("gemini_api_key"),
            configured_at=datetime.now()
        )

        # Validate at least required APIs are provided
        if not api_config.anthropic_api_key or not api_config.bright_data_api_key:
            raise HTTPException(
                status_code=400,
                detail="Anthropic and Bright Data API keys are required"
            )

        # Save API configuration
        success = await user_manager.save_api_configuration(user_data["uid"], api_config)

        if success:
            # Validate APIs
            validation_results = await user_manager.validate_user_apis(user_data["uid"])

            return {
                "success": True,
                "message": "API keys configured successfully",
                "validation": validation_results,
                "status": "active"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save API configuration")

    except Exception as e:
        logger.error(f"API configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail="API configuration failed")


@app.get("/api/auth/api-status")
async def get_api_status(user_data: Dict = Depends(get_current_user)):
    """Get user's API configuration status"""
    try:
        validation_results = await user_manager.validate_user_apis(user_data["uid"])

        return {
            "success": True,
            "api_status": validation_results,
            "configured": user_data.get("profile", {}).get("api_configured", False)
        }
    except Exception as e:
        logger.error(f"API status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get API status")


@app.get("/api/auth/api-keys")
async def get_user_api_keys(user_data: Dict = Depends(get_current_user)):
    """Get user's API keys (masked for security)"""
    try:
        api_keys = await user_manager.get_user_api_keys(user_data["uid"])

        # Mask API keys for security
        masked_keys = {}
        if api_keys:
            for key, value in api_keys.items():
                if value:
                    masked_keys[key] = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                else:
                    masked_keys[key] = None

        return {
            "success": True,
            "api_keys": masked_keys,
            "configured": bool(api_keys)
        }
    except Exception as e:
        logger.error(f"Get API keys error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get API keys")


@app.put("/api/auth/api-keys")
async def update_user_api_keys(
    request: Dict[str, Any],
    user_data: Dict = Depends(get_current_user)
):
    """Update specific API keys (CRUD Update)"""
    try:
        # Get current API keys
        current_keys = await user_manager.get_user_api_keys(user_data["uid"]) or {}

        # Update only provided keys
        updated_keys = current_keys.copy()
        for key, value in request.items():
            if key in ["anthropic_api_key", "bright_data_api_key", "openai_api_key", "gemini_api_key"]:
                if value:  # Only update if value is provided
                    updated_keys[key] = value

        # Create new API configuration
        api_config = APIConfiguration(
            anthropic_api_key=updated_keys.get("anthropic_api_key"),
            bright_data_api_key=updated_keys.get("bright_data_api_key"),
            openai_api_key=updated_keys.get("openai_api_key"),
            gemini_api_key=updated_keys.get("gemini_api_key"),
            configured_at=datetime.now()
        )

        # Save updated configuration
        success = await user_manager.save_api_configuration(user_data["uid"], api_config)

        if success:
            return {
                "success": True,
                "message": "API keys updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update API keys")

    except Exception as e:
        logger.error(f"Update API keys error: {str(e)}")
        raise HTTPException(status_code=500, detail="API keys update failed")


@app.delete("/api/auth/api-keys/{api_type}")
async def delete_api_key(
    api_type: str,
    user_data: Dict = Depends(get_current_user)
):
    """Delete specific API key (CRUD Delete)"""
    try:
        if api_type not in ["anthropic_api_key", "bright_data_api_key", "openai_api_key", "gemini_api_key"]:
            raise HTTPException(status_code=400, detail="Invalid API key type")

        # Get current API keys
        current_keys = await user_manager.get_user_api_keys(user_data["uid"]) or {}

        # Remove the specified key
        current_keys[api_type] = None

        # Create updated API configuration
        api_config = APIConfiguration(
            anthropic_api_key=current_keys.get("anthropic_api_key"),
            bright_data_api_key=current_keys.get("bright_data_api_key"),
            openai_api_key=current_keys.get("openai_api_key"),
            gemini_api_key=current_keys.get("gemini_api_key"),
            configured_at=datetime.now()
        )

        # Save updated configuration
        success = await user_manager.save_api_configuration(user_data["uid"], api_config)

        if success:
            return {
                "success": True,
                "message": f"{api_type.replace('_', ' ').title()} deleted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete API key")

    except Exception as e:
        logger.error(f"Delete API key error: {str(e)}")
        raise HTTPException(status_code=500, detail="API key deletion failed")

@app.get("/research/{tool}")
async def get_research(tool: str, user_data: Dict = Depends(get_current_user)):
    if tool not in TOOL_CONFIG:
        raise HTTPException(status_code=400, detail="Invalid tool")
    
    query = f"Run comprehensive analysis for {tool.replace('-', ' ').title()}. Fetch best-selling, low-competition, high-profit products, identify trends and gaps, and format data for UI display."
    
    try:
        result = await run_tool_agent(tool, query, user_data)
        
        # Store research history
        user_ref = db.collection("users").document(user_data["email"].replace(".", "_"))
        user_ref.collection("research_history").add({
            "tool": tool,
            "query": query,
            "result": result,
            "timestamp": datetime.now()
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Research error for {tool}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research failed for {tool}")

# Import TikTok RAG system at the top of the file
from tiktok_rag_integration import TikTokLearningService
from analytics_platform.analytics_service import AnalyticsService
from user_management import user_manager, UserStatus, APIConfiguration
from youtube_verification import youtube_verification_ai

# Initialize TikTok Learning Service
tiktok_service = TikTokLearningService()

# Initialize Analytics Service
analytics_service = AnalyticsService()

@app.post("/chat/{tool}")
async def chat(tool: str, request: ChatRequest, user_data: Dict = Depends(get_current_user)):
    """Enhanced chat endpoint with proper API routing"""
    if tool not in TOOL_CONFIG and tool != "tiktok_learning":
        raise HTTPException(status_code=400, detail="Invalid tool")

    try:
        # Special handling for TikTok Learning with Enhanced Agentic RAG
        if tool == "tiktok_learning":
            # TikTok Learning uses OWNER'S APIs (Gemini) - NO USER API CHECK NEEDED
            rag_enabled = request.context.get("rag_enabled", True) if request.context else True
            use_knowledge_graph = request.context.get("use_knowledge_graph", True) if request.context else True
            focus_category = request.context.get("focus_category") if request.context else None

            # Use OWNER'S Gemini API (system-wide, no user API required)
            result = await tiktok_service.handle_chat_request(
                message=request.message,
                user_id=user_data["email"],
                rag_enabled=rag_enabled,
                use_owner_apis=True  # Use owner's APIs, not user's
            )

            # Enhance result with system information
            if isinstance(result, dict) and "tools_used" in result:
                result["enhanced_features"] = {
                    "agentic_rag": True,
                    "knowledge_graph": use_knowledge_graph,
                    "multi_source_data": True,
                    "real_time_insights": True,
                    "owner_apis": True,  # Using owner's APIs
                    "cost_to_user": "FREE"  # Free for users
                }

            return result

        # For other tools (research tools), check user's API keys
        user_api_keys = await user_manager.get_user_api_keys(user_data["uid"])

        if not user_api_keys:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "api_configuration_required",
                    "message": "Please configure your API keys to use research tools",
                    "action": "configure_apis"
                }
            )

            # Enhance result with agentic information
            if isinstance(result, dict) and "tools_used" in result:
                result["enhanced_features"] = {
                    "agentic_rag": True,
                    "knowledge_graph": use_knowledge_graph,
                    "multi_source_data": True,
                    "real_time_insights": True
                }
        else:
            # Determine the actual tool to use
            actual_tool = tool if tool != "tiktok_learning" else "tiktok-learning"
            result = await run_tool_agent(actual_tool, request.message, user_data)
        
        # Store chat history
        user_ref = db.collection("users").document(user_data["email"].replace(".", "_"))
        user_ref.collection("chat_history").add({
            "tool": tool,
            "message": request.message,
            "response": result.get("response", result) if isinstance(result, dict) else result,
            "sources": result.get("sources", []) if isinstance(result, dict) else [],
            "confidence": result.get("confidence", 0.0) if isinstance(result, dict) else 0.0,
            "model_used": result.get("model_used", "") if isinstance(result, dict) else "",
            "timestamp": datetime.now(),
            "context": request.context
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Chat error for {tool}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed for {tool}")


# New endpoints for TikTok Agentic RAG data ingestion
@app.post("/api/tiktok-learning/ingest/youtube")
async def ingest_youtube_transcripts(
    request: Dict[str, Any],
    user_data: Dict = Depends(get_current_user)
):
    """
    Ingest YouTube video transcripts into the TikTok learning system.

    Expected request format:
    {
        "transcripts": [
            {
                "video_id": "string",
                "title": "string",
                "transcript": "string",
                "channel": "string",
                "duration": int,
                "views": int,
                "upload_date": "string",
                "tags": ["string"],
                "url": "string"
            }
        ]
    }
    """
    try:
        transcripts = request.get("transcripts", [])
        if not transcripts:
            raise HTTPException(status_code=400, detail="No transcripts provided")

        # Use the enhanced TikTok service for ingestion
        result = await tiktok_service.ingest_youtube_data(transcripts)

        # Log the ingestion for the user
        user_ref = db.collection("users").document(user_data["email"].replace(".", "_"))
        user_ref.collection("ingestion_history").add({
            "type": "youtube_transcripts",
            "count": len(transcripts),
            "timestamp": datetime.now(),
            "result": result
        })

        return {
            "success": True,
            "message": f"Successfully ingested {len(transcripts)} YouTube transcripts",
            "result": result
        }

    except Exception as e:
        logger.error(f"YouTube ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YouTube ingestion failed: {str(e)}")


@app.post("/api/tiktok-learning/ingest/facebook")
async def ingest_facebook_data(
    request: Dict[str, Any],
    user_data: Dict = Depends(get_current_user)
):
    """
    Ingest Facebook groups posts and comments into the TikTok learning system.

    Expected request format:
    {
        "facebook_data": [
            {
                "id": "string",
                "type": "post" | "comment",
                "content": "string",
                "author": "string",
                "group_name": "string",
                "likes": int,
                "comments_count": int,
                "shares": int,
                "post_date": "string",
                "engagement_score": float,
                "url": "string"
            }
        ]
    }
    """
    try:
        facebook_data = request.get("facebook_data", [])
        if not facebook_data:
            raise HTTPException(status_code=400, detail="No Facebook data provided")

        # Use the enhanced TikTok service for ingestion
        result = await tiktok_service.ingest_facebook_data(facebook_data)

        # Log the ingestion for the user
        user_ref = db.collection("users").document(user_data["email"].replace(".", "_"))
        user_ref.collection("ingestion_history").add({
            "type": "facebook_data",
            "count": len(facebook_data),
            "timestamp": datetime.now(),
            "result": result
        })

        return {
            "success": True,
            "message": f"Successfully ingested {len(facebook_data)} Facebook items",
            "result": result
        }

    except Exception as e:
        logger.error(f"Facebook ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Facebook ingestion failed: {str(e)}")


@app.get("/api/tiktok-learning/status")
async def get_tiktok_learning_status(user_data: Dict = Depends(get_current_user)):
    """Get the status of the TikTok learning system."""
    try:
        # Get system status
        status = {
            "system": "TikTok Agentic RAG Learning System v2.0",
            "features": {
                "agentic_rag": True,
                "knowledge_graph": True,
                "multi_source_data": True,
                "youtube_transcripts": True,
                "facebook_groups": True,
                "real_time_search": True,
                "gemini_api": True
            },
            "capabilities": [
                "Product hunting insights",
                "Compliance guidance",
                "Reinstatement strategies",
                "Marketing strategies",
                "Trend analysis",
                "Real-time knowledge retrieval"
            ]
        }

        return status

    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Status check failed")


@app.get("/user/status")
async def get_user_status(user_data: Dict = Depends(get_current_user)):
    return {
        "email": user_data["email"],
        "active": user_data.get("active", False),
        "verified_at": user_data.get("verified_at"),
        "created_at": user_data.get("created_at")
    }

@app.get("/tools")
async def get_available_tools():
    return {
        "tools": [
            {"id": tool_id, "name": tool_id.replace('-', ' ').title(), "platform": config["platform"]}
            for tool_id, config in TOOL_CONFIG.items()
        ]
    }

# Analytics Platform Endpoints

@app.get("/api/analytics/check-apis")
async def check_research_apis(user_data: Dict = Depends(get_current_user)):
    """Check if user has required APIs configured for research tools"""
    try:
        user_api_keys = await user_manager.get_user_api_keys(user_data["uid"])

        # Check required APIs for research tools
        required_apis = {
            "anthropic_api_key": bool(user_api_keys and user_api_keys.get("anthropic_api_key")),
            "bright_data_api_key": bool(user_api_keys and user_api_keys.get("bright_data_api_key"))
        }

        # Optional APIs
        optional_apis = {
            "openai_api_key": bool(user_api_keys and user_api_keys.get("openai_api_key")),
            "gemini_api_key": bool(user_api_keys and user_api_keys.get("gemini_api_key"))
        }

        all_required_configured = all(required_apis.values())

        return {
            "success": True,
            "apis_configured": all_required_configured,
            "required_apis": required_apis,
            "optional_apis": optional_apis,
            "missing_required": [api for api, configured in required_apis.items() if not configured],
            "can_access_research": all_required_configured
        }

    except Exception as e:
        logger.error(f"API check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check API configuration")


@app.post("/api/analytics/research")
async def perform_analytics_research(
    request: Dict[str, Any],
    user_data: Dict = Depends(get_current_user)  # Changed from get_authorized_user
):
    """
    Perform comprehensive e-commerce research using user's APIs.
    First checks if APIs are configured before proceeding.

    Request format:
    {
        "platform": "amazon|ebay|walmart|tiktok_shop",
        "search_query": "product search query",
        "analysis_types": ["best_sellers", "gap_analysis", "price_analysis"],
        "config": {
            "max_results": 50,
            "include_reviews": false,
            "export_format": "csv",
            "ui_preferences": {}
        }
    }
    """
    try:
        platform = request.get("platform")
        search_query = request.get("search_query")
        analysis_types = request.get("analysis_types", ["best_sellers"])
        config = request.get("config", {})

        if not platform or not search_query:
            raise HTTPException(status_code=400, detail="Platform and search_query are required")

        # STEP 1: Check if user has required APIs configured
        user_api_keys = await user_manager.get_user_api_keys(user_data["uid"])

        if not user_api_keys:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "api_configuration_required",
                    "message": "Please configure your API keys to use research tools",
                    "action": "configure_apis",
                    "required_apis": ["anthropic_api_key", "bright_data_api_key"]
                }
            )

        # STEP 2: Check specific required APIs
        missing_apis = []
        if not user_api_keys.get("anthropic_api_key"):
            missing_apis.append("anthropic_api_key")
        if not user_api_keys.get("bright_data_api_key"):
            missing_apis.append("bright_data_api_key")

        if missing_apis:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "missing_required_apis",
                    "message": f"Missing required API keys: {', '.join(missing_apis)}",
                    "action": "configure_apis",
                    "missing_apis": missing_apis,
                    "required_apis": ["anthropic_api_key", "bright_data_api_key"]
                }
            )

        # STEP 3: Validate API keys are working
        validation_results = await user_manager.validate_user_apis(user_data["uid"])

        if not validation_results.get("anthropic") or not validation_results.get("bright_data"):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "invalid_api_keys",
                    "message": "One or more API keys are invalid. Please check your configuration.",
                    "action": "update_apis",
                    "validation_results": validation_results
                }
            )

        # STEP 4: Proceed with research using user's APIs
        config["user_api_keys"] = user_api_keys

        research_result = await analytics_service.perform_research(
            platform=platform,
            search_query=search_query,
            analysis_types=analysis_types,
            research_config=config
        )

        return {
            "success": True,
            "research_result": research_result,
            "user_id": user_data["email"],
            "using_user_apis": True,
            "api_validation": validation_results
        }

    except HTTPException:
        # Re-raise HTTP exceptions (API configuration errors)
        raise
    except Exception as e:
        logger.error(f"Analytics research error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@app.get("/api/analytics/platforms")
async def get_supported_platforms(user_data: Dict = Depends(get_current_user)):
    """Get list of supported e-commerce platforms"""
    try:
        platforms = await analytics_service.get_supported_platforms()
        return {
            "success": True,
            "platforms": platforms
        }
    except Exception as e:
        logger.error(f"Error getting platforms: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get platforms")


@app.get("/api/analytics/analyses")
async def get_supported_analyses(user_data: Dict = Depends(get_current_user)):
    """Get list of supported analysis types"""
    try:
        analyses = await analytics_service.get_supported_analyses()
        return {
            "success": True,
            "analyses": analyses
        }
    except Exception as e:
        logger.error(f"Error getting analyses: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get analyses")


@app.get("/api/analytics/platform/{platform}/capabilities")
async def get_platform_capabilities(
    platform: str,
    user_data: Dict = Depends(get_current_user)
):
    """Get capabilities for specific platform"""
    try:
        capabilities = await analytics_service.get_platform_capabilities(platform)
        return {
            "success": True,
            "platform": platform,
            "capabilities": capabilities
        }
    except Exception as e:
        logger.error(f"Error getting platform capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get platform capabilities")


@app.post("/api/analytics/export")
async def export_research_data(
    request: Dict[str, Any],
    user_data: Dict = Depends(get_current_user)
):
    """
    Export research data in various formats (CSV, JSON, Excel).

    Request format:
    {
        "research_id": "research_123456789",
        "export_format": "csv|json|excel|parquet",
        "compression": "zip|gzip",
        "include_metadata": true,
        "include_quality_metrics": true
    }
    """
    try:
        research_id = request.get("research_id")
        export_format = request.get("export_format", "csv")

        if not research_id:
            raise HTTPException(status_code=400, detail="research_id is required")

        # For now, return a simulated export response
        # In production, you would retrieve the research data and export it

        return {
            "success": True,
            "export_info": {
                "research_id": research_id,
                "format": export_format,
                "download_url": f"/api/analytics/download/{research_id}",
                "expires_at": "2024-02-01T00:00:00Z"
            }
        }

    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/analytics/health")
async def analytics_health_check():
    """Health check for analytics platform"""
    try:
        health_status = await analytics_service.health_check()
        return health_status
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "service": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)