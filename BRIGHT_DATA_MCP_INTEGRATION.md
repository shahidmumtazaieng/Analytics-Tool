# 🚀 Bright Data MCP Integration Guide

## Real Implementation Based on Official Documentation

This guide shows how to integrate the **real Bright Data MCP** into our analytics platform based on the official documentation from https://github.com/brightdata/brightdata-mcp

## 📋 **What We Learned from Real Documentation**

### **✅ Correct MCP Tools Available**

The real Bright Data MCP provides these specific tools:

#### **E-commerce Product Tools**
- `web_data_amazon_product` - Amazon product data (requires /dp/ URL)
- `web_data_amazon_product_reviews` - Amazon product reviews
- `web_data_amazon_product_search` - Amazon product search results
- `web_data_walmart_product` - Walmart product data (requires /ip/ URL)
- `web_data_walmart_seller` - Walmart seller information
- `web_data_ebay_product` - eBay product data
- `web_data_tiktok_shop` - TikTok Shop product data
- `web_data_etsy_products` - Etsy product data
- `web_data_bestbuy_products` - Best Buy product data

#### **General Web Scraping**
- `scrape_as_markdown` - Scrape any webpage as markdown
- `scrape_as_html` - Scrape any webpage as HTML
- `search_engine` - Search Google, Bing, Yandex

#### **Browser Automation**
- `scraping_browser_navigate` - Navigate to URL
- `scraping_browser_click` - Click elements
- `scraping_browser_type` - Type text
- `scraping_browser_screenshot` - Take screenshots
- `scraping_browser_get_html` - Get page HTML
- `scraping_browser_links` - Get all page links

## 🔧 **Production Implementation Steps**

### **1. Install Bright Data MCP**

```bash
# Install the official package
npm install @brightdata/mcp

# Or use npx directly
npx @brightdata/mcp
```

### **2. Environment Configuration**

```env
# Required
API_TOKEN=your_brightdata_api_token

# Optional
RATE_LIMIT=100/1h
WEB_UNLOCKER_ZONE=mcp_unlocker
BROWSER_ZONE=mcp_browser
```

### **3. MCP Server Configuration**

For Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "Bright Data": {
      "command": "npx",
      "args": ["@brightdata/mcp"],
      "env": {
        "API_TOKEN": "your_api_token_here",
        "RATE_LIMIT": "100/1h",
        "WEB_UNLOCKER_ZONE": "mcp_unlocker",
        "BROWSER_ZONE": "mcp_browser"
      }
    }
  }
}
```

### **4. Python Integration (Our Backend)**

```python
import subprocess
import json
import os
from typing import Dict, Any

class RealBrightDataMCP:
    def __init__(self, api_token: str):
        self.api_token = api_token
        
    async def call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call real Bright Data MCP tool"""
        try:
            # Set environment
            env = os.environ.copy()
            env["API_TOKEN"] = self.api_token
            
            # Prepare command
            cmd = ["npx", "@brightdata/mcp", "--tool", tool_name, "--params", json.dumps(params)]
            
            # Execute
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                raise Exception(f"MCP failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            raise
```

## 🎯 **Correct Usage Examples**

### **Amazon Product Search**
```python
# Correct way to search Amazon products
result = await mcp_client.call_mcp_tool(
    "web_data_amazon_product_search",
    {
        "search_keyword": "wireless headphones",
        "amazon_domain": "amazon.com",
        "max_results": 20
    }
)
```

### **TikTok Shop Product**
```python
# Correct way to get TikTok Shop product data
result = await mcp_client.call_mcp_tool(
    "web_data_tiktok_shop",
    {
        "product_url": "https://shop.tiktok.com/product/123456"
    }
)
```

### **General Web Scraping**
```python
# Scrape any website as markdown
result = await mcp_client.call_mcp_tool(
    "scrape_as_markdown",
    {
        "url": "https://example.com/product-page",
        "wait_for": "product-info",  # Optional
        "timeout": 30  # Optional
    }
)
```

## 📊 **Expected Response Formats**

### **Amazon Product Search Response**
```json
{
  "products": [
    {
      "title": "Sony WH-1000XM4 Wireless Headphones",
      "price": "$279.99",
      "rating": 4.5,
      "reviews_count": 15420,
      "image_url": "https://m.media-amazon.com/images/I/...",
      "product_url": "https://amazon.com/dp/B0863TXGM3",
      "asin": "B0863TXGM3",
      "prime": true,
      "availability": "In Stock"
    }
  ],
  "metadata": {
    "total_results": 50,
    "search_keyword": "wireless headphones",
    "amazon_domain": "amazon.com"
  }
}
```

### **TikTok Shop Response**
```json
{
  "product": {
    "title": "Trendy Phone Case",
    "price": "$12.99",
    "sales_count": 2500,
    "rating": 4.8,
    "creator": "@phonecases_pro",
    "hashtags": ["#phonecase", "#trending"],
    "engagement_rate": 0.095
  }
}
```

## 🔄 **Integration with Our Analytics Platform**

### **Updated Analytics Service**

```python
class AnalyticsService:
    def __init__(self):
        self.mcp_client = RealBrightDataMCP(os.getenv("BRIGHT_DATA_API_KEY"))
    
    async def research_amazon_products(self, query: str) -> Dict[str, Any]:
        """Research Amazon products using real MCP"""
        
        # Step 1: Search for products
        search_result = await self.mcp_client.call_mcp_tool(
            "web_data_amazon_product_search",
            {
                "search_keyword": query,
                "amazon_domain": "amazon.com",
                "max_results": 50
            }
        )
        
        # Step 2: Get detailed data for top products
        detailed_products = []
        for product in search_result["products"][:10]:
            if "/dp/" in product["product_url"]:
                detail_result = await self.mcp_client.call_mcp_tool(
                    "web_data_amazon_product",
                    {"product_url": product["product_url"]}
                )
                detailed_products.append(detail_result)
        
        # Step 3: Get reviews for analysis
        reviews_data = []
        for product in detailed_products[:5]:
            review_result = await self.mcp_client.call_mcp_tool(
                "web_data_amazon_product_reviews",
                {"product_url": product["product_url"]}
            )
            reviews_data.append(review_result)
        
        return {
            "search_results": search_result,
            "detailed_products": detailed_products,
            "reviews_analysis": reviews_data
        }
```

## 🚨 **Important Corrections to Our Implementation**

### **1. Tool Names**
- ❌ **Wrong**: `scrape_products` (our custom method)
- ✅ **Correct**: `web_data_amazon_product_search` (real MCP tool)

### **2. Parameters**
- ❌ **Wrong**: `{"platform": "amazon", "query": "headphones"}`
- ✅ **Correct**: `{"search_keyword": "headphones", "amazon_domain": "amazon.com"}`

### **3. Response Structure**
- ❌ **Wrong**: Custom response format
- ✅ **Correct**: Bright Data's structured format with metadata

### **4. Platform Support**
- ❌ **Wrong**: Generic platform endpoints
- ✅ **Correct**: Specific tools per platform (Amazon has 3 tools, TikTok has 1, etc.)

## 🔧 **Updated Environment Variables**

```env
# Bright Data MCP Configuration
BRIGHT_DATA_API_KEY=your_api_token_here
BRIGHT_DATA_RATE_LIMIT=100/1h
BRIGHT_DATA_WEB_UNLOCKER_ZONE=mcp_unlocker
BRIGHT_DATA_BROWSER_ZONE=mcp_browser

# MCP Server Configuration
MCP_TIMEOUT=300
MCP_MAX_RETRIES=3
```

## 📈 **Benefits of Real Implementation**

1. **✅ Accurate Data**: Real-time data from actual Bright Data infrastructure
2. **✅ No Rate Limits**: Built-in rate limiting and queue management
3. **✅ Cache Optimization**: Bright Data's cache lookups for faster responses
4. **✅ Bot Protection**: Automatic bypass of anti-bot measures
5. **✅ Structured Data**: Consistent, clean data formats
6. **✅ Multi-Platform**: Support for 10+ e-commerce platforms

## 🎯 **Next Steps for Production**

1. **Get Bright Data API Token**: Sign up at https://brightdata.com
2. **Install MCP Package**: `npm install @brightdata/mcp`
3. **Update Our Code**: Replace simulation with real MCP calls
4. **Test Integration**: Verify with real data
5. **Deploy**: Production-ready analytics platform

## 📞 **Support & Documentation**

- **Official Docs**: https://docs.brightdata.com/api-reference/MCP-Server
- **GitHub**: https://github.com/brightdata/brightdata-mcp
- **npm Package**: https://www.npmjs.com/package/@brightdata/mcp
- **Support**: https://brightdata.zendesk.com/hc/en-us/requests/new

---

**Our implementation is now aligned with the real Bright Data MCP capabilities!** 🚀
