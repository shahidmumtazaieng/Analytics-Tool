import os
import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages
from firebase_admin import credentials, initialize_app, auth, firestore
import firebase_admin
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Firebase
cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH"))
initialize_app(cred)
db = firestore.client()

# FastAPI app
app = FastAPI(title="E-commerce Product Hunter SaaS")

# OAuth2 for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Pydantic models
class User(BaseModel):
    email: str
    bright_data_api_key: str
    openai_api_key: str

class ChatRequest(BaseModel):
    tool: str
    message: str

class ImageVerification(BaseModel):
    youtube_url: str
    screenshot_path: str

# Dependency to verify user and get Firestore data
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        decoded = auth.verify_id_token(token)
        user_id = decoded["uid"]
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict()
        if not user_data:
            raise HTTPException(status_code=401, detail="User not found")
        if not user_data.get("active"):
            raise HTTPException(status_code=403, detail="User inactive: Must be subscribed to YouTube channel")
        return user_data
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

# Initialize Bright Data client
async def get_bright_data_client(api_key: str):
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

# Tool configuration
TOOL_CONFIG = {
    "global-research": {
        "platform": "global",
        "tools": ["web_data_amazon_product_search", "web_data_walmart_product", "web_data_ebay_product", 
                  "web_data_homedepot_products", "web_data_zara_products", "web_data_etsy_products", 
                  "web_data_bestbuy_products", "web_data_tiktok_shop"],
        "prompt": """You are a global e-commerce research analyst. Use all available tools to:
        - Aggregate best-selling products across platforms.
        - Identify product gaps (products on one platform but not others).
        - Rank categories by sales, profitability, and competition.
        Return JSON with: product list (name, category, platform, price, sales, sellers, reviews, gap_status), category rankings, trends."""
    },
    "product-gap-finder": {
        "platform": "global",
        "tools": ["web_data_amazon_product_search", "web_data_walmart_product", "web_data_ebay_product", 
                  "web_data_homedepot_products", "web_data_zara_products", "web_data_etsy_products", 
                  "web_data_bestbuy_products", "web_data_tiktok_shop"],
        "prompt": """You are a product gap analyst. Use all available tools to:
        - Identify products selling well on one platform but absent or underperforming on others.
        - Provide gap details (e.g., 'Available on Amazon, missing on eBay').
        Return JSON with: product list (name, category, platform, price, sales, sellers, reviews, gap_status)."""
    },
    "amazon-best-sellers": {
        "platform": "amazon",
        "tools": ["web_data_amazon_product_search"],
        "prompt": """You are an Amazon research analyst. Use web_data_amazon_product_search to:
        - Identify best-selling products (high sales volume, reviews).
        - Find low-competition products (<10 sellers, <500 reviews).
        - Identify high-profit products (3x markup after 30% fees, 20% costs).
        Return JSON with: product list (name, category, price, sales, sellers, reviews), top categories."""
    },
    "amazon-product-gaps": {
        "platform": "amazon",
        "tools": ["web_data_amazon_product_search", "web_data_walmart_product", "web_data_ebay_product", 
                  "web_data_homedepot_products", "web_data_zara_products", "web_data_etsy_products", 
                  "web_data_bestbuy_products", "web_data_tiktok_shop"],
        "prompt": """You are an Amazon gap analyst. Use web_data_amazon_product_search and other tools to:
        - Identify products selling well on Amazon but absent or underperforming on other platforms.
        Return JSON with: product list (name, category, platform, price, sales, sellers, reviews, gap_status)."""
    },
    "ebay-best-sellers": {
        "platform": "ebay",
        "tools": ["web_data_ebay_product"],
        "prompt": """You are an eBay research analyst. Use web_data_ebay_product to:
        - Identify best-selling products (high sales volume, reviews).
        - Find low-competition products (<10 sellers, <500 reviews).
        - Identify high-profit products (3x markup after 30% fees, 20% costs).
        Return JSON with: product list (name, category, price, sales, sellers, reviews), top categories."""
    },
    "walmart-best-sellers": {
        "platform": "walmart",
        "tools": ["web_data_walmart_product"],
        "prompt": """You are a Walmart research analyst. Use web_data_walmart_product to:
        - Identify best-selling products (high sales volume, reviews).
        - Find low-competition products (<10 sellers, <500 reviews).
        - Identify high-profit products (3x markup after 30% fees, 20% costs).
        Return JSON with: product list (name, category, price, sales, sellers, reviews), top categories."""
    },
    "homedepot-best-sellers": {
        "platform": "homedepot",
        "tools": ["web_data_homedepot_products"],
        "prompt": """You are a Home Depot research analyst. Use web_data_homedepot_products to:
        - Identify best-selling products (high sales volume, reviews).
        - Find low-competition products (<10 sellers, <500 reviews).
        - Identify high-profit products (3x markup after 30% fees, 20% costs).
        Return JSON with: product list (name, category, price, sales, sellers, reviews), top categories."""
    },
    "zara-best-sellers": {
        "platform": "zara",
        "tools": ["web_data_zara_products"],
        "prompt": """You are a Zara research analyst. Use web_data_zara_products to:
        - Identify best-selling products (high sales volume, reviews).
        - Find low-competition products (<10 sellers, <500 reviews).
        - Identify high-profit products (3x markup after 30% fees, 20% costs).
        Return JSON with: product list (name, category, price, sales, sellers, reviews), top categories."""
    },
    "etsy-best-sellers": {
        "platform": "etsy",
        "tools": ["web_data_etsy_products"],
        "prompt": """You are an Etsy research analyst. Use web_data_etsy_products to:
        - Identify best-selling products (high sales volume, reviews).
        - Find low-competition products (<10 sellers, <500 reviews).
        - Identify high-profit products (3x markup after 30% fees, 20% costs).
        Return JSON with: product list (name, category, price, sales, sellers, reviews), top categories."""
    },
    "bestbuy-best-sellers": {
        "platform": "bestbuy",
        "tools": ["web_data_bestbuy_products"],
        "prompt": """You are a Best Buy research analyst. Use web_data_bestbuy_products to:
        - Identify best-selling products (high sales volume, reviews).
        - Find low-competition products (<10 sellers, <500 reviews).
        - Identify high-profit products (3x markup after 30% fees, 20% costs).
        Return JSON with: product list (name, category, price, sales, sellers, reviews), top categories."""
    },
    "tiktok-best-sellers": {
        "platform": "tiktok",
        "tools": ["web_data_tiktok_shop"],
        "prompt": """You are a TikTok Shop research analyst. Use web_data_tiktok_shop to:
        - Identify best-selling products (high sales volume, reviews).
        - Find low-competition products (<10 sellers, <500 reviews).
        - Identify high-profit products (3x markup after 30% fees, 20% costs).
        Return JSON with: product list (name, category, price, sales, sellers, reviews), top categories."""
    }
}

# Simulate image recognition (replace with Google Cloud Vision)
async def verify_youtube_subscription(youtube_url: str, screenshot_path: str) -> bool:
    logger.info(f"Verifying YouTube subscription for URL: {youtube_url}, screenshot: {screenshot_path}")
    # Placeholder: Implement Google Cloud Vision to check channel name and "Subscribed" status
    return True  # Mock response for demo

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
    
    client = await get_bright_data_client(user_data["bright_data_api_key"])
    tools = await client.get_tools()
    model = init_chat_model(model="openai:gpt-4.1", api_key=user_data["openai_api_key"])

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
        - Profitability: Calculate margins (3x markup after 30% fees, 20% costs).
        - Competition: Score (<10 sellers, <500 reviews = low competition).
        - Demand: Score based on sales volume and trends.
        - Gaps: Confirm products missing on specific platforms (if applicable).
        Return JSON with: product list (name, category, platform, price, profitability_score, competition_score, demand_score, gap_status), analytics summary.""",
        name="analytics_agent"
    )

    # Chat agent
    chat_agent = create_react_agent(
        model,
        tools,
        prompt=f"""You are a chat assistant for the {tool.replace('-', ' ').title()} tool. Answer queries about products, categories, trends, or gaps using the latest data. Return concise, actionable responses in JSON or plain text as requested.""",
        name=f"{tool}_chat_agent"
    )

    # TikTok learning chatbot with RAG (simulated)
    tiktok_learning_agent = create_react_agent(
        model,
        tools,
        prompt="""You are a TikTok learning chatbot with RAG, fine-tuned on TikTok Shop data and e-commerce trends. Use web_data_tiktok_shop and RAG context to answer questions about trending products, selling strategies, or TikTok Shop insights. Return JSON with answer and sources.""",
        name="tiktok_learning_agent"
    )

    # UI formatter agent
    ui_formatter_agent = create_react_agent(
        model,
        tools,
        prompt="""You are a UI data formatter. Format data for a React frontend in JSON:
        - Product list: name, category, platform, price, profitability_score, competition_score, demand_score, gap_status.
        - Category rankings: name, total_sales, avg_profitability, avg_competition, top_products, gap_opportunities.
        - Metadata: sort_options (profitability, demand, competition), filter_options (platform, category, gap_status).""",
        name="ui_formatter_agent"
    )

    # Supervisor
    agents = [research_agent, analytics_agent, chat_agent, tiktok_learning_agent, ui_formatter_agent]
    supervisor = create_supervisor(
        model=init_chat_model("openai:gpt-4.1", api_key=user_data["openai_api_key"]),
        agents=agents,
        prompt=(
            f"You are a supervisor for the {tool.replace('-', ' ').title()} tool:\n"
            f"- {tool}_research_agent: Handles research for {tool}.\n"
            f"- analytics_agent: Analyzes profitability, competition, demand, gaps.\n"
            f"- {tool}_chat_agent: Answers queries for {tool}.\n"
            f"- tiktok_learning_agent: Handles TikTok learning queries with RAG.\n"
            f"- ui_formatter_agent: Formats data for UI.\n"
            f"Assign tasks based on query. Do not call agents in parallel."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()

    result = []
    for chunk in supervisor.stream({"messages": [{"role": "user", "content": query}]}):
        result.extend(pretty_print_messages(chunk, last_message=True))
    return result[-1]["content"] if result else {}

# API Endpoints
@app.post("/login")
async def login(user: User):
    try:
        user_ref = db.collection("users").document(user.email.replace(".", "_"))
        user_ref.set({
            "email": user.email,
            "bright_data_api_key": user.bright_data_api_key,
            "openai_api_key": user.openai_api_key,
            "active": False,
            "created_at": datetime.now()
        })
        return {"message": "User registered, proceed to image verification"}
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=400, detail="Registration failed")

@app.post("/verify-youtube")
async def verify_youtube(verification: ImageVerification, user_data: Dict = Depends(get_current_user)):
    is_subscribed = await verify_youtube_subscription(verification.youtube_url, verification.screenshot_path)
    if is_subscribed:
        user_ref = db.collection("users").document(user_data["email"].replace(".", "_"))
        user_ref.update({"active": True})
        return {"message": "Subscription verified, access granted"}
    raise HTTPException(status_code=403, detail="Not subscribed to YouTube channel")

@app.get("/research/{tool}")
async def get_research(tool: str, user_data: Dict = Depends(get_current_user)):
    if tool not in TOOL_CONFIG:
        raise HTTPException(status_code=400, detail="Invalid tool")
    query = f"Run analysis for {tool.replace('-', ' ').title()}. Fetch best-selling, low-competition, high-profit products, identify trends and gaps, and format data for UI."
    try:
        result = await run_tool_agent(tool, query, user_data)
        return result
    except Exception as e:
        logger.error(f"Research error for {tool}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research failed for {tool}")

@app.post("/chat/{tool}")
async def chat(tool: str, request: ChatRequest, user_data: Dict = Depends(get_current_user)):
    if tool not in TOOL_CONFIG and tool != "tiktok_learning":
        raise HTTPException(status_code=400, detail="Invalid tool")
    try:
        result = await run_tool_agent(request.message if tool != "tiktok_learning" else request.message, 
                                     tool if tool != "tiktok_learning" else "tiktok", user_data)
        # Store chat history
        user_ref = db.collection("users").document(user_data["email"].replace(".", "_"))
        user_ref.collection("chat_history").add({
            "tool": tool,
            "message": request.message,
            "response": result,
            "timestamp": datetime.now()
        })
        return result
    except Exception as e:
        logger.error(f"Chat error for {tool}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed for {tool}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)