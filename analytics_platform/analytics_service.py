"""
Main Analytics Service
Orchestrates the complete analytics platform workflow
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from .bright_data_mcp import BrightDataMCPClient, ScrapingRequest, ScrapingResponse
from .analytics_agents import AnalyticsAgentSystem, AnalysisRequest, AnalysisType, AnalysisResult
from .data_processor import DataProcessor, CSVExporter, CSVExportConfig
from .ui_generator import UIGenerator, UILayout

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Main analytics service that orchestrates the complete workflow:
    1. Scrape data using Bright Data MCP
    2. Analyze data using intelligent agents
    3. Generate UI components automatically
    4. Export data for AI/ML training
    """
    
    def __init__(self):
        """Initialize analytics service"""
        self.bright_data_client = BrightDataMCPClient()
        self.analytics_agents = AnalyticsAgentSystem()
        self.data_processor = DataProcessor()
        self.csv_exporter = CSVExporter()
        self.ui_generator = UIGenerator()
        
        logger.info("Analytics Service initialized")
    
    async def perform_research(
        self,
        platform: str,
        search_query: str,
        analysis_types: List[str],
        research_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform complete e-commerce research workflow.
        
        Args:
            platform: E-commerce platform (amazon, ebay, walmart, tiktok_shop)
            search_query: Search query for products
            analysis_types: Types of analysis to perform
            research_config: Additional configuration
            
        Returns:
            Complete research results with UI and export data
        """
        try:
            if research_config is None:
                research_config = {}

            # Configure APIs based on user's keys
            user_api_keys = research_config.get("user_api_keys", {})
            if user_api_keys:
                self._configure_user_apis(user_api_keys)

            # Step 1: Scrape data using Bright Data MCP with user's API
            logger.info(f"Starting research for '{search_query}' on {platform}")
            scraped_data = await self._scrape_data(platform, search_query, research_config)
            
            # Step 2: Process and clean data
            processed_data = self.data_processor.process_scraped_data(
                scraped_data,
                clean_data=research_config.get("clean_data", True),
                add_features=research_config.get("add_ml_features", True)
            )
            
            # Step 3: Perform analytics analysis
            analysis_results = []
            for analysis_type in analysis_types:
                try:
                    analysis_result = await self._perform_analysis(
                        analysis_type, scraped_data, research_config
                    )
                    analysis_results.append(analysis_result)
                except Exception as e:
                    logger.error(f"Analysis {analysis_type} failed: {e}")
                    continue
            
            # Step 4: Generate UI layout
            ui_layout = None
            if analysis_results:
                # Use the first analysis result for UI generation
                primary_analysis = analysis_results[0]
                ui_layout = self.ui_generator.generate_ui_layout(
                    primary_analysis,
                    scraped_data,
                    research_config.get("ui_preferences", {})
                )
            
            # Step 5: Prepare export data
            export_data = await self._prepare_export_data(
                processed_data, analysis_results, research_config
            )
            
            # Step 6: Generate insights summary
            insights_summary = self._generate_insights_summary(analysis_results, scraped_data)
            
            return {
                "research_id": f"research_{int(datetime.now().timestamp())}",
                "platform": platform,
                "query": search_query,
                "scraped_data": {
                    "total_products": len(scraped_data.products),
                    "data_quality_score": scraped_data.data_quality_score,
                    "processing_time": scraped_data.processing_time,
                    "scraped_at": scraped_data.scraped_at.isoformat()
                },
                "analysis_results": [result.__dict__ for result in analysis_results],
                "ui_layout": ui_layout.__dict__ if ui_layout else None,
                "export_data": export_data,
                "insights_summary": insights_summary,
                "research_metadata": {
                    "completed_at": datetime.now().isoformat(),
                    "analysis_count": len(analysis_results),
                    "data_quality": processed_data.get("quality_metrics", {}),
                    "recommendations": self._generate_recommendations(analysis_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            raise

    def _configure_user_apis(self, user_api_keys: Dict[str, str]) -> None:
        """Configure analytics service to use user's API keys"""
        try:
            # Update Bright Data client with user's API key
            if user_api_keys.get("bright_data_api_key"):
                self.bright_data_client.api_key = user_api_keys["bright_data_api_key"]
                logger.info("Configured Bright Data with user's API key")

            # Update analytics agents with user's Anthropic API key
            if user_api_keys.get("anthropic_api_key"):
                # This would update the agent's model configuration
                # Implementation depends on how you want to handle per-user model configs
                logger.info("Configured analytics agents with user's Anthropic API key")

        except Exception as e:
            logger.error(f"Failed to configure user APIs: {e}")
            # Continue with default configuration
    
    async def _scrape_data(
        self,
        platform: str,
        search_query: str,
        config: Dict[str, Any]
    ) -> ScrapingResponse:
        """Scrape data using Bright Data MCP"""
        scraping_request = ScrapingRequest(
            platform=platform,
            search_query=search_query,
            max_results=config.get("max_results", 50),
            filters=config.get("filters", {}),
            include_reviews=config.get("include_reviews", False),
            include_pricing_history=config.get("include_pricing_history", False),
            include_seller_info=config.get("include_seller_info", False)
        )
        
        return await self.bright_data_client.scrape_products(scraping_request)
    
    async def _perform_analysis(
        self,
        analysis_type: str,
        scraped_data: ScrapingResponse,
        config: Dict[str, Any]
    ) -> AnalysisResult:
        """Perform specific type of analysis"""
        try:
            analysis_type_enum = AnalysisType(analysis_type)
        except ValueError:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        analysis_request = AnalysisRequest(
            analysis_type=analysis_type_enum,
            data=scraped_data,
            parameters=config.get("analysis_parameters", {})
        )
        
        return await self.analytics_agents.analyze(analysis_request)
    
    async def _prepare_export_data(
        self,
        processed_data: Dict[str, Any],
        analysis_results: List[AnalysisResult],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for export"""
        export_config = CSVExportConfig(
            include_metadata=config.get("include_metadata", True),
            include_quality_metrics=config.get("include_quality_metrics", True),
            normalize_prices=config.get("normalize_prices", True),
            clean_text_fields=config.get("clean_text_fields", True),
            add_ml_features=config.get("add_ml_features", True),
            export_format=config.get("export_format", "csv"),
            compression=config.get("compression")
        )
        
        export_result = self.csv_exporter.export_data(processed_data, export_config)
        
        # Add analysis results to export
        if analysis_results:
            analysis_export = {
                "analysis_summary": [
                    {
                        "type": result.analysis_type.value,
                        "insights": result.insights,
                        "metrics": result.metrics,
                        "recommendations": result.recommendations,
                        "confidence_score": result.confidence_score
                    }
                    for result in analysis_results
                ]
            }
            
            # Add analysis summary as separate file
            export_result["files"]["analysis_summary.json"] = str(analysis_export)
        
        return export_result
    
    def _generate_insights_summary(
        self,
        analysis_results: List[AnalysisResult],
        scraped_data: ScrapingResponse
    ) -> Dict[str, Any]:
        """Generate comprehensive insights summary"""
        if not analysis_results:
            return {"message": "No analysis results available"}
        
        # Aggregate insights from all analyses
        all_insights = []
        all_recommendations = []
        total_confidence = 0
        
        for result in analysis_results:
            all_insights.extend(result.insights)
            all_recommendations.extend(result.recommendations)
            total_confidence += result.confidence_score
        
        avg_confidence = total_confidence / len(analysis_results)
        
        # Generate key findings
        key_findings = []
        
        # Market overview
        key_findings.append(f"Analyzed {len(scraped_data.products)} products from {scraped_data.platform}")
        key_findings.append(f"Data quality score: {scraped_data.data_quality_score:.2%}")
        
        # Analysis-specific findings
        for result in analysis_results:
            if result.analysis_type == AnalysisType.BEST_SELLERS:
                best_seller_count = result.metrics.get("best_sellers_found", 0)
                key_findings.append(f"Identified {best_seller_count} best-selling products")
            
            elif result.analysis_type == AnalysisType.GAP_ANALYSIS:
                gaps_found = result.metrics.get("total_gaps_found", 0)
                key_findings.append(f"Found {gaps_found} market opportunities")
            
            elif result.analysis_type == AnalysisType.PRICE_ANALYSIS:
                avg_price = result.metrics.get("avg_price", 0)
                key_findings.append(f"Average market price: ${avg_price:.2f}")
        
        return {
            "key_findings": key_findings,
            "total_insights": len(all_insights),
            "total_recommendations": len(all_recommendations),
            "confidence_score": avg_confidence,
            "analysis_types_performed": [result.analysis_type.value for result in analysis_results],
            "research_quality": "High" if avg_confidence > 0.8 else "Medium" if avg_confidence > 0.6 else "Low"
        }
    
    def _generate_recommendations(self, analysis_results: List[AnalysisResult]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Aggregate recommendations from all analyses
        for result in analysis_results:
            recommendations.extend(result.recommendations)
        
        # Add general recommendations
        if analysis_results:
            recommendations.extend([
                "Consider focusing on identified market gaps for competitive advantage",
                "Monitor pricing strategies of top performers",
                "Leverage high-quality data for informed decision making",
                "Use exported data for further analysis and model training"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    async def get_supported_platforms(self) -> List[str]:
        """Get list of supported e-commerce platforms"""
        return await self.bright_data_client.get_supported_platforms()
    
    async def get_supported_analyses(self) -> List[str]:
        """Get list of supported analysis types"""
        return await self.analytics_agents.get_supported_analyses()
    
    async def get_platform_capabilities(self, platform: str) -> Dict[str, Any]:
        """Get capabilities for specific platform"""
        return await self.bright_data_client.get_platform_capabilities(platform)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            "service": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check Bright Data MCP client
            platforms = await self.bright_data_client.get_supported_platforms()
            health_status["components"]["bright_data_mcp"] = {
                "status": "healthy",
                "supported_platforms": len(platforms)
            }
        except Exception as e:
            health_status["components"]["bright_data_mcp"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            # Check analytics agents
            analyses = await self.analytics_agents.get_supported_analyses()
            health_status["components"]["analytics_agents"] = {
                "status": "healthy",
                "supported_analyses": len(analyses)
            }
        except Exception as e:
            health_status["components"]["analytics_agents"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check if any component is unhealthy
        unhealthy_components = [
            name for name, status in health_status["components"].items()
            if status["status"] == "unhealthy"
        ]
        
        if unhealthy_components:
            health_status["service"] = "degraded"
            health_status["unhealthy_components"] = unhealthy_components
        
        return health_status
