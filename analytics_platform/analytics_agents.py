"""
Intelligent Analytics Agents System
Specialized agents for understanding and processing e-commerce data
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

from ..agentic_rag_system.agent.providers import get_mcp_model
from .bright_data_mcp import ScrapingResponse

logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Types of analytics analysis"""
    BEST_SELLERS = "best_sellers"
    GAP_ANALYSIS = "gap_analysis"
    PRICE_ANALYSIS = "price_analysis"
    TREND_ANALYSIS = "trend_analysis"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    DEMAND_FORECAST = "demand_forecast"


class UIComponentType(str, Enum):
    """Types of UI components to generate"""
    PRODUCT_GRID = "product_grid"
    COMPARISON_TABLE = "comparison_table"
    PRICE_CHART = "price_chart"
    TREND_GRAPH = "trend_graph"
    ANALYTICS_CARDS = "analytics_cards"
    FILTER_PANEL = "filter_panel"


@dataclass
class AnalysisRequest:
    """Request for analytics analysis"""
    analysis_type: AnalysisType
    data: ScrapingResponse
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class UISection:
    """UI section configuration"""
    component_type: UIComponentType
    title: str
    data: Dict[str, Any]
    config: Dict[str, Any]
    priority: int = 1


@dataclass
class AnalysisResult:
    """Result of analytics analysis"""
    analysis_type: AnalysisType
    insights: List[str]
    metrics: Dict[str, Any]
    ui_sections: List[UISection]
    csv_data: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float


class AnalyticsAgentDependencies(BaseModel):
    """Dependencies for analytics agents"""
    user_id: str
    analysis_preferences: Dict[str, Any] = {}
    ui_preferences: Dict[str, Any] = {}


# Initialize the analytics agent system
analytics_agent = Agent(
    get_mcp_model(),
    deps_type=AnalyticsAgentDependencies,
    system_prompt="""You are an expert e-commerce analytics agent with deep understanding of market research, 
    competitive analysis, and data visualization. You specialize in:

    1. **Best Seller Analysis**: Identifying top-performing products and trends
    2. **Gap Analysis**: Finding market opportunities and underserved niches  
    3. **Price Analysis**: Competitive pricing strategies and optimization
    4. **Trend Analysis**: Market trends and demand patterns
    5. **Competitor Analysis**: Competitive landscape and positioning
    6. **UI Generation**: Creating appropriate data visualizations and interfaces

    Your responses should be data-driven, actionable, and include specific recommendations 
    for e-commerce success. Always consider both the business opportunity and the technical 
    implementation for UI components."""
)


@analytics_agent.tool
async def analyze_best_sellers(
    ctx: RunContext[AnalyticsAgentDependencies],
    scraped_data: Dict[str, Any],
    min_sales_threshold: int = 100,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Analyze best-selling products from scraped data.
    
    Args:
        scraped_data: Raw scraped product data
        min_sales_threshold: Minimum sales count to consider
        top_n: Number of top products to return
    
    Returns:
        Best seller analysis with insights and UI configuration
    """
    try:
        products = scraped_data.get("products", [])
        
        # Sort by sales indicators (reviews_count, rating, etc.)
        best_sellers = []
        for product in products:
            sales_score = 0
            
            # Calculate sales score based on available metrics
            if "reviews_count" in product:
                sales_score += product["reviews_count"] * 0.4
            if "rating" in product:
                sales_score += product["rating"] * 20
            if "sales_count" in product:
                sales_score += product["sales_count"] * 0.6
            
            product["sales_score"] = sales_score
            
            if sales_score >= min_sales_threshold:
                best_sellers.append(product)
        
        # Sort by sales score
        best_sellers.sort(key=lambda x: x["sales_score"], reverse=True)
        top_sellers = best_sellers[:top_n]
        
        # Generate insights
        insights = []
        if top_sellers:
            avg_price = sum(p.get("price", 0) for p in top_sellers) / len(top_sellers)
            avg_rating = sum(p.get("rating", 0) for p in top_sellers) / len(top_sellers)
            
            insights.extend([
                f"Top {len(top_sellers)} best sellers identified",
                f"Average price of best sellers: ${avg_price:.2f}",
                f"Average rating of best sellers: {avg_rating:.1f}/5.0",
                f"Price range: ${min(p.get('price', 0) for p in top_sellers):.2f} - ${max(p.get('price', 0) for p in top_sellers):.2f}"
            ])
        
        # Generate UI sections
        ui_sections = [
            {
                "component_type": "product_grid",
                "title": "Best Selling Products",
                "data": {"products": top_sellers},
                "config": {"columns": 4, "show_metrics": True},
                "priority": 1
            },
            {
                "component_type": "analytics_cards",
                "title": "Best Seller Metrics",
                "data": {
                    "total_products": len(top_sellers),
                    "avg_price": avg_price if top_sellers else 0,
                    "avg_rating": avg_rating if top_sellers else 0,
                    "price_range": f"${min(p.get('price', 0) for p in top_sellers):.2f} - ${max(p.get('price', 0) for p in top_sellers):.2f}" if top_sellers else "N/A"
                },
                "config": {"layout": "horizontal"},
                "priority": 2
            }
        ]
        
        return {
            "analysis_type": "best_sellers",
            "insights": insights,
            "top_products": top_sellers,
            "ui_sections": ui_sections,
            "metrics": {
                "total_analyzed": len(products),
                "best_sellers_found": len(top_sellers),
                "avg_sales_score": sum(p["sales_score"] for p in top_sellers) / len(top_sellers) if top_sellers else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Best seller analysis failed: {e}")
        return {"error": str(e), "analysis_type": "best_sellers"}


@analytics_agent.tool
async def analyze_market_gaps(
    ctx: RunContext[AnalyticsAgentDependencies],
    scraped_data: Dict[str, Any],
    price_ranges: List[tuple] = None,
    category_focus: str = None
) -> Dict[str, Any]:
    """
    Identify market gaps and opportunities.
    
    Args:
        scraped_data: Raw scraped product data
        price_ranges: Price ranges to analyze for gaps
        category_focus: Specific category to focus on
    
    Returns:
        Gap analysis with opportunities and recommendations
    """
    try:
        products = scraped_data.get("products", [])
        
        if price_ranges is None:
            price_ranges = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 500)]
        
        # Analyze price distribution
        price_distribution = {f"${r[0]}-${r[1]}": 0 for r in price_ranges}
        rating_gaps = {}
        feature_gaps = []
        
        for product in products:
            price = product.get("price", 0)
            rating = product.get("rating", 0)
            
            # Count products in each price range
            for min_price, max_price in price_ranges:
                if min_price <= price <= max_price:
                    range_key = f"${min_price}-${max_price}"
                    price_distribution[range_key] += 1
                    
                    # Track rating gaps
                    if range_key not in rating_gaps:
                        rating_gaps[range_key] = []
                    rating_gaps[range_key].append(rating)
        
        # Identify gaps
        gaps = []
        opportunities = []
        
        # Price gaps (ranges with few products)
        for price_range, count in price_distribution.items():
            if count < 3:  # Threshold for gap
                gaps.append(f"Low competition in {price_range} price range ({count} products)")
                opportunities.append({
                    "type": "price_gap",
                    "range": price_range,
                    "competition_level": "low",
                    "product_count": count
                })
        
        # Rating gaps (price ranges with low average ratings)
        for price_range, ratings in rating_gaps.items():
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                if avg_rating < 4.0:
                    gaps.append(f"Quality gap in {price_range} range (avg rating: {avg_rating:.1f})")
                    opportunities.append({
                        "type": "quality_gap",
                        "range": price_range,
                        "avg_rating": avg_rating,
                        "improvement_potential": 5.0 - avg_rating
                    })
        
        # Generate recommendations
        recommendations = []
        if opportunities:
            recommendations.extend([
                "Focus on underserved price ranges with low competition",
                "Improve product quality in ranges with low ratings",
                "Consider premium positioning in gaps above $100",
                "Target budget-conscious consumers in sub-$25 gaps"
            ])
        
        # UI sections for gap analysis
        ui_sections = [
            {
                "component_type": "comparison_table",
                "title": "Price Range Analysis",
                "data": {"price_distribution": price_distribution},
                "config": {"highlight_gaps": True},
                "priority": 1
            },
            {
                "component_type": "analytics_cards",
                "title": "Market Opportunities",
                "data": {"opportunities": opportunities},
                "config": {"show_priority": True},
                "priority": 2
            }
        ]
        
        return {
            "analysis_type": "gap_analysis",
            "insights": gaps,
            "opportunities": opportunities,
            "recommendations": recommendations,
            "ui_sections": ui_sections,
            "metrics": {
                "total_gaps_found": len(gaps),
                "high_opportunity_ranges": len([o for o in opportunities if o.get("competition_level") == "low"]),
                "avg_market_saturation": sum(price_distribution.values()) / len(price_distribution)
            }
        }
        
    except Exception as e:
        logger.error(f"Gap analysis failed: {e}")
        return {"error": str(e), "analysis_type": "gap_analysis"}


@analytics_agent.tool
async def analyze_pricing_strategy(
    ctx: RunContext[AnalyticsAgentDependencies],
    scraped_data: Dict[str, Any],
    target_margin: float = 0.3
) -> Dict[str, Any]:
    """
    Analyze pricing strategies and competitive positioning.
    
    Args:
        scraped_data: Raw scraped product data
        target_margin: Target profit margin for recommendations
    
    Returns:
        Pricing analysis with strategy recommendations
    """
    try:
        products = scraped_data.get("products", [])
        
        if not products:
            return {"error": "No products to analyze", "analysis_type": "price_analysis"}
        
        # Price statistics
        prices = [p.get("price", 0) for p in products if p.get("price", 0) > 0]
        
        if not prices:
            return {"error": "No valid prices found", "analysis_type": "price_analysis"}
        
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        median_price = sorted(prices)[len(prices) // 2]
        
        # Price segments
        segments = {
            "budget": [p for p in prices if p < avg_price * 0.7],
            "mid_range": [p for p in prices if avg_price * 0.7 <= p <= avg_price * 1.3],
            "premium": [p for p in prices if p > avg_price * 1.3]
        }
        
        # Competitive analysis
        insights = [
            f"Average market price: ${avg_price:.2f}",
            f"Price range: ${min_price:.2f} - ${max_price:.2f}",
            f"Median price: ${median_price:.2f}",
            f"Budget segment: {len(segments['budget'])} products (< ${avg_price * 0.7:.2f})",
            f"Mid-range segment: {len(segments['mid_range'])} products",
            f"Premium segment: {len(segments['premium'])} products (> ${avg_price * 1.3:.2f})"
        ]
        
        # Pricing recommendations
        recommendations = []
        if len(segments['budget']) > len(segments['premium']):
            recommendations.append("Market is budget-focused - consider competitive pricing")
        if len(segments['premium']) > len(segments['budget']):
            recommendations.append("Premium market opportunity - focus on quality differentiation")
        
        recommended_price = median_price * (1 + target_margin)
        recommendations.append(f"Recommended entry price: ${recommended_price:.2f} (based on {target_margin*100}% margin)")
        
        # UI sections
        ui_sections = [
            {
                "component_type": "price_chart",
                "title": "Price Distribution Analysis",
                "data": {
                    "prices": prices,
                    "segments": segments,
                    "statistics": {
                        "avg": avg_price,
                        "median": median_price,
                        "min": min_price,
                        "max": max_price
                    }
                },
                "config": {"chart_type": "histogram"},
                "priority": 1
            }
        ]
        
        return {
            "analysis_type": "price_analysis",
            "insights": insights,
            "recommendations": recommendations,
            "ui_sections": ui_sections,
            "metrics": {
                "avg_price": avg_price,
                "price_volatility": (max_price - min_price) / avg_price,
                "recommended_price": recommended_price,
                "market_segments": {k: len(v) for k, v in segments.items()}
            }
        }
        
    except Exception as e:
        logger.error(f"Price analysis failed: {e}")
        return {"error": str(e), "analysis_type": "price_analysis"}


class AnalyticsAgentSystem:
    """Main analytics agent system coordinator"""
    
    def __init__(self):
        """Initialize the analytics agent system"""
        self.agent = analytics_agent
        self.supported_analyses = [
            AnalysisType.BEST_SELLERS,
            AnalysisType.GAP_ANALYSIS,
            AnalysisType.PRICE_ANALYSIS,
            AnalysisType.TREND_ANALYSIS,
            AnalysisType.COMPETITOR_ANALYSIS
        ]
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform analytics analysis using appropriate agent tools.
        
        Args:
            request: Analysis request configuration
            
        Returns:
            Comprehensive analysis result
        """
        try:
            # Prepare agent dependencies
            deps = AnalyticsAgentDependencies(
                user_id="analytics_user",
                analysis_preferences=request.parameters
            )
            
            # Route to appropriate analysis tool
            if request.analysis_type == AnalysisType.BEST_SELLERS:
                result = await analyze_best_sellers(
                    ctx=None,  # Will be provided by agent
                    scraped_data=request.data.__dict__,
                    min_sales_threshold=request.parameters.get("min_sales_threshold", 100),
                    top_n=request.parameters.get("top_n", 20)
                )
            elif request.analysis_type == AnalysisType.GAP_ANALYSIS:
                result = await analyze_market_gaps(
                    ctx=None,
                    scraped_data=request.data.__dict__,
                    price_ranges=request.parameters.get("price_ranges"),
                    category_focus=request.parameters.get("category_focus")
                )
            elif request.analysis_type == AnalysisType.PRICE_ANALYSIS:
                result = await analyze_pricing_strategy(
                    ctx=None,
                    scraped_data=request.data.__dict__,
                    target_margin=request.parameters.get("target_margin", 0.3)
                )
            else:
                raise ValueError(f"Unsupported analysis type: {request.analysis_type}")
            
            # Convert to AnalysisResult
            return AnalysisResult(
                analysis_type=request.analysis_type,
                insights=result.get("insights", []),
                metrics=result.get("metrics", {}),
                ui_sections=[UISection(**section) for section in result.get("ui_sections", [])],
                csv_data=self._prepare_csv_data(result),
                recommendations=result.get("recommendations", []),
                confidence_score=0.85  # Default confidence
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _prepare_csv_data(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare data for CSV export"""
        csv_data = []
        
        # Extract products if available
        if "top_products" in analysis_result:
            csv_data = analysis_result["top_products"]
        elif "opportunities" in analysis_result:
            csv_data = analysis_result["opportunities"]
        
        return csv_data
    
    async def get_supported_analyses(self) -> List[str]:
        """Get list of supported analysis types"""
        return [analysis.value for analysis in self.supported_analyses]
