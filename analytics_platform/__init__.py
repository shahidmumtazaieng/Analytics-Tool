"""
Analytics Platform with Bright Data MCP Integration
Advanced e-commerce research and analytics platform
"""

from .bright_data_mcp import BrightDataMCPClient
from .analytics_agents import AnalyticsAgentSystem
from .data_processor import DataProcessor, CSVExporter
from .ui_generator import UIGenerator

__version__ = "1.0.0"
__author__ = "E-commerce Analytics Team"

__all__ = [
    "BrightDataMCPClient",
    "AnalyticsAgentSystem", 
    "DataProcessor",
    "CSVExporter",
    "UIGenerator"
]
