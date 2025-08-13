"""
Dynamic UI Interface Generation System
Automatically creates analytics interfaces based on scraped data structure
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .analytics_agents import UISection, UIComponentType, AnalysisResult
from .bright_data_mcp import ScrapingResponse

logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Types of charts for data visualization"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"


class LayoutType(str, Enum):
    """Layout types for UI components"""
    GRID = "grid"
    LIST = "list"
    CARDS = "cards"
    TABLE = "table"
    CAROUSEL = "carousel"


@dataclass
class UIComponent:
    """Individual UI component configuration"""
    id: str
    type: UIComponentType
    title: str
    data: Dict[str, Any]
    config: Dict[str, Any]
    styling: Dict[str, Any]
    interactions: List[str]
    priority: int = 1


@dataclass
class UILayout:
    """Complete UI layout configuration"""
    components: List[UIComponent]
    layout_config: Dict[str, Any]
    responsive_breakpoints: Dict[str, Any]
    theme: str = "default"


class UIGenerator:
    """
    Dynamic UI generator that creates appropriate interfaces
    based on data structure and analysis results.
    """
    
    def __init__(self):
        """Initialize UI generator"""
        self.component_templates = {
            UIComponentType.PRODUCT_GRID: self._generate_product_grid,
            UIComponentType.COMPARISON_TABLE: self._generate_comparison_table,
            UIComponentType.PRICE_CHART: self._generate_price_chart,
            UIComponentType.TREND_GRAPH: self._generate_trend_graph,
            UIComponentType.ANALYTICS_CARDS: self._generate_analytics_cards,
            UIComponentType.FILTER_PANEL: self._generate_filter_panel
        }
        
        self.default_styling = {
            "colors": {
                "primary": "#3B82F6",
                "secondary": "#10B981", 
                "accent": "#F59E0B",
                "danger": "#EF4444",
                "success": "#22C55E",
                "warning": "#F59E0B"
            },
            "spacing": {
                "xs": "0.25rem",
                "sm": "0.5rem",
                "md": "1rem",
                "lg": "1.5rem",
                "xl": "2rem"
            },
            "typography": {
                "font_family": "Inter, system-ui, sans-serif",
                "heading_sizes": ["2xl", "xl", "lg", "md"],
                "body_size": "sm"
            }
        }
    
    def generate_ui_layout(
        self,
        analysis_result: AnalysisResult,
        scraped_data: ScrapingResponse,
        layout_preferences: Dict[str, Any] = None
    ) -> UILayout:
        """
        Generate complete UI layout based on analysis results.
        
        Args:
            analysis_result: Results from analytics analysis
            scraped_data: Original scraped data
            layout_preferences: User layout preferences
            
        Returns:
            Complete UI layout configuration
        """
        try:
            components = []
            
            # Generate components from analysis UI sections
            for ui_section in analysis_result.ui_sections:
                component = self._create_component_from_section(ui_section, scraped_data)
                if component:
                    components.append(component)
            
            # Add additional components based on data structure
            additional_components = self._generate_additional_components(scraped_data, analysis_result)
            components.extend(additional_components)
            
            # Sort components by priority
            components.sort(key=lambda x: x.priority)
            
            # Generate layout configuration
            layout_config = self._generate_layout_config(components, layout_preferences)
            
            # Generate responsive breakpoints
            responsive_breakpoints = self._generate_responsive_config(components)
            
            return UILayout(
                components=components,
                layout_config=layout_config,
                responsive_breakpoints=responsive_breakpoints,
                theme=layout_preferences.get("theme", "default") if layout_preferences else "default"
            )
            
        except Exception as e:
            logger.error(f"UI layout generation failed: {e}")
            raise
    
    def _create_component_from_section(
        self,
        ui_section: UISection,
        scraped_data: ScrapingResponse
    ) -> Optional[UIComponent]:
        """Create UI component from analysis UI section"""
        try:
            generator_func = self.component_templates.get(ui_section.component_type)
            if not generator_func:
                logger.warning(f"No generator for component type: {ui_section.component_type}")
                return None
            
            # Generate component using appropriate template
            component_data = generator_func(ui_section.data, ui_section.config, scraped_data)
            
            return UIComponent(
                id=f"{ui_section.component_type.value}_{hash(ui_section.title)}",
                type=ui_section.component_type,
                title=ui_section.title,
                data=component_data["data"],
                config=component_data["config"],
                styling=component_data.get("styling", self.default_styling),
                interactions=component_data.get("interactions", []),
                priority=ui_section.priority
            )
            
        except Exception as e:
            logger.error(f"Component creation failed for {ui_section.component_type}: {e}")
            return None
    
    def _generate_product_grid(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        scraped_data: ScrapingResponse
    ) -> Dict[str, Any]:
        """Generate product grid component"""
        products = data.get("products", [])
        
        # Enhance products with display data
        enhanced_products = []
        for product in products:
            enhanced_product = {
                "id": product.get("id", ""),
                "title": product.get("title", "Unknown Product"),
                "price": f"${product.get('price', 0):.2f}",
                "rating": product.get("rating", 0),
                "reviews_count": product.get("reviews_count", 0),
                "image_url": product.get("image_url", "/placeholder-product.jpg"),
                "product_url": product.get("product_url", "#"),
                "availability": product.get("availability", "unknown"),
                "badges": self._generate_product_badges(product),
                "metrics": self._generate_product_metrics(product)
            }
            enhanced_products.append(enhanced_product)
        
        return {
            "data": {
                "products": enhanced_products,
                "total_count": len(enhanced_products),
                "platform": scraped_data.platform
            },
            "config": {
                "columns": config.get("columns", 4),
                "show_metrics": config.get("show_metrics", True),
                "enable_sorting": True,
                "enable_filtering": True,
                "pagination": {
                    "enabled": len(enhanced_products) > 20,
                    "page_size": 20
                },
                "card_style": "elevated",
                "image_aspect_ratio": "1:1"
            },
            "styling": {
                **self.default_styling,
                "grid": {
                    "gap": "1rem",
                    "min_card_width": "250px"
                }
            },
            "interactions": ["sort", "filter", "view_details", "compare"]
        }
    
    def _generate_comparison_table(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        scraped_data: ScrapingResponse
    ) -> Dict[str, Any]:
        """Generate comparison table component"""
        
        # Handle different data structures
        if "price_distribution" in data:
            # Price range comparison
            table_data = []
            for price_range, count in data["price_distribution"].items():
                table_data.append({
                    "price_range": price_range,
                    "product_count": count,
                    "market_share": f"{(count / sum(data['price_distribution'].values()) * 100):.1f}%",
                    "competition_level": "Low" if count < 3 else "Medium" if count < 8 else "High"
                })
        else:
            # Generic table data
            table_data = data.get("items", [])
        
        return {
            "data": {
                "rows": table_data,
                "columns": self._generate_table_columns(table_data),
                "summary": self._generate_table_summary(table_data)
            },
            "config": {
                "sortable": True,
                "filterable": True,
                "exportable": True,
                "highlight_gaps": config.get("highlight_gaps", False),
                "row_actions": ["view", "analyze"],
                "pagination": len(table_data) > 50
            },
            "styling": {
                **self.default_styling,
                "table": {
                    "stripe_rows": True,
                    "hover_effect": True,
                    "border_style": "minimal"
                }
            },
            "interactions": ["sort", "filter", "export", "row_select"]
        }
    
    def _generate_price_chart(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        scraped_data: ScrapingResponse
    ) -> Dict[str, Any]:
        """Generate price chart component"""
        prices = data.get("prices", [])
        segments = data.get("segments", {})
        statistics = data.get("statistics", {})
        
        chart_type = config.get("chart_type", "histogram")
        
        if chart_type == "histogram":
            # Price distribution histogram
            chart_data = self._create_price_histogram(prices)
        else:
            # Price trend line chart
            chart_data = self._create_price_trend(prices)
        
        return {
            "data": {
                "chart_data": chart_data,
                "statistics": statistics,
                "segments": segments,
                "insights": self._generate_price_insights(prices, statistics)
            },
            "config": {
                "chart_type": chart_type,
                "interactive": True,
                "show_legend": True,
                "show_tooltips": True,
                "export_options": ["png", "svg", "pdf"],
                "zoom_enabled": True
            },
            "styling": {
                **self.default_styling,
                "chart": {
                    "height": "400px",
                    "background": "white",
                    "border_radius": "8px"
                }
            },
            "interactions": ["zoom", "pan", "hover", "export"]
        }
    
    def _generate_trend_graph(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        scraped_data: ScrapingResponse
    ) -> Dict[str, Any]:
        """Generate trend graph component"""
        # Simulate trend data based on scraped data
        trend_data = self._create_trend_data(data, scraped_data)
        
        return {
            "data": {
                "series": trend_data,
                "time_range": "30d",
                "metrics": ["price", "popularity", "availability"]
            },
            "config": {
                "chart_type": "line",
                "multi_series": True,
                "time_navigation": True,
                "forecast_enabled": True,
                "annotations": True
            },
            "styling": {
                **self.default_styling,
                "chart": {
                    "height": "350px",
                    "line_width": 2,
                    "point_size": 4
                }
            },
            "interactions": ["time_range", "series_toggle", "forecast"]
        }
    
    def _generate_analytics_cards(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        scraped_data: ScrapingResponse
    ) -> Dict[str, Any]:
        """Generate analytics cards component"""
        cards = []
        
        # Generate cards based on data structure
        for key, value in data.items():
            if key.startswith("_"):
                continue  # Skip internal fields
            
            card = {
                "id": key,
                "title": self._format_card_title(key),
                "value": self._format_card_value(value),
                "icon": self._get_card_icon(key),
                "trend": self._calculate_card_trend(key, value),
                "color": self._get_card_color(key)
            }
            cards.append(card)
        
        return {
            "data": {
                "cards": cards,
                "layout": config.get("layout", "grid")
            },
            "config": {
                "columns": 4,
                "responsive": True,
                "animated": True,
                "clickable": True
            },
            "styling": {
                **self.default_styling,
                "cards": {
                    "border_radius": "12px",
                    "shadow": "medium",
                    "padding": "1.5rem"
                }
            },
            "interactions": ["click", "hover"]
        }
    
    def _generate_filter_panel(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any],
        scraped_data: ScrapingResponse
    ) -> Dict[str, Any]:
        """Generate filter panel component"""
        products = scraped_data.products
        
        # Generate filter options based on product data
        filters = []
        
        # Price range filter
        prices = [p.get("price", 0) for p in products if p.get("price", 0) > 0]
        if prices:
            filters.append({
                "type": "range",
                "field": "price",
                "label": "Price Range",
                "min": min(prices),
                "max": max(prices),
                "step": 1
            })
        
        # Rating filter
        ratings = [p.get("rating", 0) for p in products if p.get("rating", 0) > 0]
        if ratings:
            filters.append({
                "type": "range",
                "field": "rating",
                "label": "Minimum Rating",
                "min": 0,
                "max": 5,
                "step": 0.5
            })
        
        # Availability filter
        availability_options = list(set(p.get("availability", "") for p in products))
        if availability_options:
            filters.append({
                "type": "select",
                "field": "availability",
                "label": "Availability",
                "options": availability_options
            })
        
        return {
            "data": {
                "filters": filters,
                "active_filters": {},
                "filter_count": 0
            },
            "config": {
                "collapsible": True,
                "position": "sidebar",
                "apply_mode": "instant",
                "reset_enabled": True
            },
            "styling": {
                **self.default_styling,
                "panel": {
                    "width": "300px",
                    "background": "gray.50",
                    "border_radius": "8px"
                }
            },
            "interactions": ["filter_change", "reset", "collapse"]
        }
    
    def _generate_additional_components(
        self,
        scraped_data: ScrapingResponse,
        analysis_result: AnalysisResult
    ) -> List[UIComponent]:
        """Generate additional components based on data characteristics"""
        additional_components = []
        
        # Add export component if we have data
        if scraped_data.products:
            export_component = UIComponent(
                id="export_panel",
                type=UIComponentType.FILTER_PANEL,  # Reuse for export panel
                title="Export Data",
                data={
                    "export_formats": ["CSV", "JSON", "Excel"],
                    "total_records": len(scraped_data.products),
                    "data_quality": analysis_result.metrics.get("data_quality", 0.9)
                },
                config={
                    "position": "bottom",
                    "export_enabled": True
                },
                styling=self.default_styling,
                interactions=["export"],
                priority=10
            )
            additional_components.append(export_component)
        
        return additional_components
    
    # Helper methods for component generation
    def _generate_product_badges(self, product: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate badges for product display"""
        badges = []
        
        if product.get("prime"):
            badges.append({"text": "Prime", "color": "blue"})
        
        if product.get("rating", 0) >= 4.5:
            badges.append({"text": "Top Rated", "color": "green"})
        
        if product.get("sales_count", 0) > 1000:
            badges.append({"text": "Best Seller", "color": "orange"})
        
        availability = product.get("availability", "").lower()
        if "limited" in availability:
            badges.append({"text": "Limited Stock", "color": "red"})
        
        return badges
    
    def _generate_product_metrics(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metrics for product display"""
        return {
            "popularity_score": product.get("popularity_score", 0),
            "quality_score": product.get("quality_score", 0),
            "competitiveness_score": product.get("competitiveness_score", 0)
        }
    
    def _generate_table_columns(self, table_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate table column configuration"""
        if not table_data:
            return []
        
        columns = []
        sample_row = table_data[0]
        
        for key in sample_row.keys():
            column = {
                "field": key,
                "header": self._format_column_header(key),
                "sortable": True,
                "filterable": True,
                "type": self._detect_column_type(sample_row[key])
            }
            columns.append(column)
        
        return columns
    
    def _generate_table_summary(self, table_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate table summary statistics"""
        return {
            "total_rows": len(table_data),
            "columns": len(table_data[0].keys()) if table_data else 0,
            "last_updated": "now"
        }
    
    def _create_price_histogram(self, prices: List[float]) -> Dict[str, Any]:
        """Create price histogram data"""
        if not prices:
            return {"bins": [], "counts": []}
        
        # Create price bins
        min_price = min(prices)
        max_price = max(prices)
        bin_count = min(20, len(set(prices)))
        bin_width = (max_price - min_price) / bin_count
        
        bins = []
        counts = []
        
        for i in range(bin_count):
            bin_start = min_price + (i * bin_width)
            bin_end = bin_start + bin_width
            count = sum(1 for p in prices if bin_start <= p < bin_end)
            
            bins.append(f"${bin_start:.0f}-${bin_end:.0f}")
            counts.append(count)
        
        return {
            "labels": bins,
            "data": counts,
            "type": "histogram"
        }
    
    def _create_price_trend(self, prices: List[float]) -> Dict[str, Any]:
        """Create price trend data"""
        # Simulate trend over time
        return {
            "labels": [f"Day {i+1}" for i in range(len(prices[:30]))],
            "data": prices[:30],
            "type": "line"
        }
    
    def _generate_price_insights(self, prices: List[float], statistics: Dict[str, Any]) -> List[str]:
        """Generate insights from price data"""
        if not prices:
            return []
        
        insights = []
        avg_price = statistics.get("avg", sum(prices) / len(prices))
        
        insights.append(f"Average price: ${avg_price:.2f}")
        insights.append(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        
        if len(set(prices)) > 1:
            volatility = (max(prices) - min(prices)) / avg_price
            if volatility > 0.5:
                insights.append("High price volatility detected")
            else:
                insights.append("Stable pricing across products")
        
        return insights
    
    def _create_trend_data(self, data: Dict[str, Any], scraped_data: ScrapingResponse) -> List[Dict[str, Any]]:
        """Create trend data for visualization"""
        # Simulate trend data based on scraped data
        return [
            {
                "name": "Average Price",
                "data": [45, 52, 48, 61, 55, 67, 59],
                "color": "#3B82F6"
            },
            {
                "name": "Popularity Score",
                "data": [0.6, 0.7, 0.65, 0.8, 0.75, 0.85, 0.82],
                "color": "#10B981"
            }
        ]
    
    def _format_card_title(self, key: str) -> str:
        """Format card title from key"""
        return key.replace("_", " ").title()
    
    def _format_card_value(self, value: Any) -> str:
        """Format card value for display"""
        if isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            return str(value)
    
    def _get_card_icon(self, key: str) -> str:
        """Get appropriate icon for card"""
        icon_map = {
            "total_products": "package",
            "avg_price": "dollar-sign",
            "avg_rating": "star",
            "price_range": "trending-up"
        }
        return icon_map.get(key, "info")
    
    def _calculate_card_trend(self, key: str, value: Any) -> Dict[str, Any]:
        """Calculate trend for card (simulated)"""
        return {
            "direction": "up",
            "percentage": 5.2,
            "period": "vs last week"
        }
    
    def _get_card_color(self, key: str) -> str:
        """Get color theme for card"""
        color_map = {
            "total_products": "blue",
            "avg_price": "green",
            "avg_rating": "yellow",
            "price_range": "purple"
        }
        return color_map.get(key, "gray")
    
    def _format_column_header(self, key: str) -> str:
        """Format column header from key"""
        return key.replace("_", " ").title()
    
    def _detect_column_type(self, value: Any) -> str:
        """Detect column data type"""
        if isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, bool):
            return "boolean"
        else:
            return "text"
    
    def _generate_layout_config(
        self,
        components: List[UIComponent],
        preferences: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate layout configuration"""
        return {
            "type": "responsive_grid",
            "columns": 12,
            "gap": "1rem",
            "padding": "1rem",
            "max_width": "1200px",
            "center": True
        }
    
    def _generate_responsive_config(self, components: List[UIComponent]) -> Dict[str, Any]:
        """Generate responsive breakpoint configuration"""
        return {
            "mobile": {"max_width": "768px", "columns": 1},
            "tablet": {"max_width": "1024px", "columns": 2},
            "desktop": {"min_width": "1025px", "columns": 3}
        }
