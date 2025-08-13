"""
Data Processing and CSV Export System
For AI engineers and model training data preparation
"""

import csv
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from io import StringIO, BytesIO
import zipfile

from .bright_data_mcp import ScrapingResponse
from .analytics_agents import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    freshness_score: float
    overall_score: float
    issues: List[str]


@dataclass
class CSVExportConfig:
    """Configuration for CSV export"""
    include_metadata: bool = True
    include_quality_metrics: bool = True
    normalize_prices: bool = True
    clean_text_fields: bool = True
    add_ml_features: bool = True
    export_format: str = "csv"  # csv, json, parquet
    compression: Optional[str] = None  # gzip, zip


class DataProcessor:
    """
    Advanced data processor for e-commerce analytics data.
    Optimized for AI/ML model training and research.
    """
    
    def __init__(self):
        """Initialize data processor"""
        self.quality_thresholds = {
            "completeness": 0.8,
            "accuracy": 0.85,
            "consistency": 0.9,
            "freshness": 0.95
        }
    
    def process_scraped_data(
        self,
        scraped_data: ScrapingResponse,
        clean_data: bool = True,
        add_features: bool = True
    ) -> Dict[str, Any]:
        """
        Process raw scraped data for analysis and export.
        
        Args:
            scraped_data: Raw scraped data from Bright Data
            clean_data: Whether to clean and normalize data
            add_features: Whether to add ML-ready features
            
        Returns:
            Processed data with quality metrics
        """
        try:
            products = scraped_data.products.copy()
            
            if clean_data:
                products = self._clean_products(products)
            
            if add_features:
                products = self._add_ml_features(products)
            
            # Calculate quality metrics
            quality_metrics = self._assess_data_quality(products)
            
            return {
                "products": products,
                "metadata": {
                    "platform": scraped_data.platform,
                    "query": scraped_data.query,
                    "total_results": len(products),
                    "scraped_at": scraped_data.scraped_at.isoformat(),
                    "processing_time": scraped_data.processing_time,
                    "processed_at": datetime.now().isoformat()
                },
                "quality_metrics": quality_metrics.__dict__,
                "schema": self._generate_schema(products)
            }
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise
    
    def _clean_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and normalize product data"""
        cleaned_products = []
        
        for product in products:
            cleaned_product = {}
            
            # Clean text fields
            for key, value in product.items():
                if isinstance(value, str):
                    # Remove extra whitespace, normalize encoding
                    cleaned_value = value.strip()
                    cleaned_value = cleaned_value.replace('\n', ' ').replace('\r', ' ')
                    cleaned_value = ' '.join(cleaned_value.split())  # Normalize whitespace
                    cleaned_product[key] = cleaned_value
                elif isinstance(value, (int, float)):
                    # Ensure numeric values are valid
                    if value >= 0:  # Assuming no negative prices/ratings
                        cleaned_product[key] = value
                else:
                    cleaned_product[key] = value
            
            # Normalize price format
            if "price" in cleaned_product:
                price = cleaned_product["price"]
                if isinstance(price, str):
                    # Extract numeric value from price string
                    import re
                    price_match = re.search(r'[\d,]+\.?\d*', price.replace(',', ''))
                    if price_match:
                        cleaned_product["price"] = float(price_match.group())
                    else:
                        cleaned_product["price"] = 0.0
            
            # Normalize rating to 0-5 scale
            if "rating" in cleaned_product:
                rating = cleaned_product["rating"]
                if isinstance(rating, (int, float)) and rating > 5:
                    # Assume it's on a 0-100 scale, convert to 0-5
                    cleaned_product["rating"] = rating / 20.0
            
            cleaned_products.append(cleaned_product)
        
        return cleaned_products
    
    def _add_ml_features(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add machine learning ready features"""
        enhanced_products = []
        
        # Calculate global statistics for feature engineering
        prices = [p.get("price", 0) for p in products if p.get("price", 0) > 0]
        ratings = [p.get("rating", 0) for p in products if p.get("rating", 0) > 0]
        
        avg_price = sum(prices) / len(prices) if prices else 0
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        for product in products:
            enhanced_product = product.copy()
            
            # Price-based features
            price = product.get("price", 0)
            if price > 0:
                enhanced_product["price_category"] = self._categorize_price(price, avg_price)
                enhanced_product["price_zscore"] = (price - avg_price) / (max(prices) - min(prices)) if len(set(prices)) > 1 else 0
                enhanced_product["is_premium"] = price > avg_price * 1.5
                enhanced_product["is_budget"] = price < avg_price * 0.7
            
            # Rating-based features
            rating = product.get("rating", 0)
            if rating > 0:
                enhanced_product["rating_category"] = self._categorize_rating(rating)
                enhanced_product["above_avg_rating"] = rating > avg_rating
            
            # Review-based features
            reviews_count = product.get("reviews_count", 0)
            if reviews_count > 0:
                enhanced_product["review_volume_category"] = self._categorize_review_volume(reviews_count)
                enhanced_product["popularity_score"] = self._calculate_popularity_score(rating, reviews_count)
            
            # Text-based features
            title = product.get("title", "")
            if title:
                enhanced_product["title_length"] = len(title)
                enhanced_product["title_word_count"] = len(title.split())
                enhanced_product["has_brand_in_title"] = self._detect_brand_in_title(title)
            
            # Availability features
            availability = product.get("availability", "")
            enhanced_product["in_stock"] = availability.lower() in ["in_stock", "available", "in stock"]
            
            # Platform-specific features
            if "prime" in product:
                enhanced_product["has_prime"] = bool(product["prime"])
            
            if "sales_count" in product:
                sales = product["sales_count"]
                enhanced_product["sales_category"] = self._categorize_sales(sales)
            
            # Composite scores
            enhanced_product["quality_score"] = self._calculate_quality_score(enhanced_product)
            enhanced_product["competitiveness_score"] = self._calculate_competitiveness_score(enhanced_product, avg_price, avg_rating)
            
            enhanced_products.append(enhanced_product)
        
        return enhanced_products
    
    def _categorize_price(self, price: float, avg_price: float) -> str:
        """Categorize price relative to average"""
        if price < avg_price * 0.5:
            return "very_low"
        elif price < avg_price * 0.8:
            return "low"
        elif price < avg_price * 1.2:
            return "medium"
        elif price < avg_price * 2.0:
            return "high"
        else:
            return "very_high"
    
    def _categorize_rating(self, rating: float) -> str:
        """Categorize rating"""
        if rating >= 4.5:
            return "excellent"
        elif rating >= 4.0:
            return "good"
        elif rating >= 3.5:
            return "average"
        elif rating >= 3.0:
            return "below_average"
        else:
            return "poor"
    
    def _categorize_review_volume(self, reviews_count: int) -> str:
        """Categorize review volume"""
        if reviews_count >= 1000:
            return "high"
        elif reviews_count >= 100:
            return "medium"
        elif reviews_count >= 10:
            return "low"
        else:
            return "very_low"
    
    def _categorize_sales(self, sales_count: int) -> str:
        """Categorize sales volume"""
        if sales_count >= 10000:
            return "viral"
        elif sales_count >= 1000:
            return "popular"
        elif sales_count >= 100:
            return "moderate"
        else:
            return "low"
    
    def _calculate_popularity_score(self, rating: float, reviews_count: int) -> float:
        """Calculate popularity score based on rating and review volume"""
        if rating == 0 or reviews_count == 0:
            return 0.0
        
        # Weighted score: rating quality + review volume (log scale)
        import math
        rating_score = rating / 5.0  # Normalize to 0-1
        volume_score = min(math.log10(reviews_count + 1) / 4.0, 1.0)  # Log scale, cap at 1
        
        return (rating_score * 0.7) + (volume_score * 0.3)
    
    def _detect_brand_in_title(self, title: str) -> bool:
        """Detect if title contains brand indicators"""
        brand_indicators = ["brand", "®", "™", "official", "authentic"]
        title_lower = title.lower()
        return any(indicator in title_lower for indicator in brand_indicators)
    
    def _calculate_quality_score(self, product: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        score = 0.0
        factors = 0
        
        # Rating factor
        if "rating" in product and product["rating"] > 0:
            score += (product["rating"] / 5.0) * 0.4
            factors += 0.4
        
        # Review volume factor
        if "reviews_count" in product and product["reviews_count"] > 0:
            import math
            volume_score = min(math.log10(product["reviews_count"] + 1) / 4.0, 1.0)
            score += volume_score * 0.3
            factors += 0.3
        
        # Availability factor
        if product.get("in_stock", False):
            score += 0.2
            factors += 0.2
        
        # Brand factor
        if product.get("has_brand_in_title", False):
            score += 0.1
            factors += 0.1
        
        return score / factors if factors > 0 else 0.0
    
    def _calculate_competitiveness_score(self, product: Dict[str, Any], avg_price: float, avg_rating: float) -> float:
        """Calculate competitiveness score"""
        score = 0.0
        
        # Price competitiveness (lower price = higher score)
        price = product.get("price", 0)
        if price > 0 and avg_price > 0:
            price_score = max(0, 1 - (price / avg_price - 0.5))  # Optimal around 50% below average
            score += price_score * 0.4
        
        # Quality competitiveness
        rating = product.get("rating", 0)
        if rating > 0 and avg_rating > 0:
            quality_score = rating / 5.0
            score += quality_score * 0.6
        
        return min(score, 1.0)
    
    def _assess_data_quality(self, products: List[Dict[str, Any]]) -> DataQualityMetrics:
        """Assess data quality metrics"""
        if not products:
            return DataQualityMetrics(0, 0, 0, 0, 0, ["No products to assess"])
        
        issues = []
        
        # Completeness: percentage of products with required fields
        required_fields = ["title", "price"]
        complete_products = 0
        for product in products:
            if all(field in product and product[field] for field in required_fields):
                complete_products += 1
        
        completeness_score = complete_products / len(products)
        
        # Accuracy: check for reasonable values
        accurate_products = 0
        for product in products:
            is_accurate = True
            
            # Price should be positive
            if "price" in product:
                price = product["price"]
                if not isinstance(price, (int, float)) or price <= 0:
                    is_accurate = False
            
            # Rating should be 0-5
            if "rating" in product:
                rating = product["rating"]
                if not isinstance(rating, (int, float)) or rating < 0 or rating > 5:
                    is_accurate = False
            
            if is_accurate:
                accurate_products += 1
        
        accuracy_score = accurate_products / len(products)
        
        # Consistency: check for consistent data types and formats
        consistency_score = 0.9  # Simplified for now
        
        # Freshness: based on scraping timestamp
        freshness_score = 1.0  # Assume fresh data
        
        # Overall score
        overall_score = (completeness_score + accuracy_score + consistency_score + freshness_score) / 4
        
        # Add issues
        if completeness_score < self.quality_thresholds["completeness"]:
            issues.append(f"Low completeness: {completeness_score:.2%}")
        if accuracy_score < self.quality_thresholds["accuracy"]:
            issues.append(f"Low accuracy: {accuracy_score:.2%}")
        
        return DataQualityMetrics(
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            freshness_score=freshness_score,
            overall_score=overall_score,
            issues=issues
        )
    
    def _generate_schema(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data schema for documentation"""
        if not products:
            return {}
        
        schema = {}
        sample_product = products[0]
        
        for key, value in sample_product.items():
            schema[key] = {
                "type": type(value).__name__,
                "description": self._get_field_description(key),
                "example": value
            }
        
        return schema
    
    def _get_field_description(self, field_name: str) -> str:
        """Get description for field"""
        descriptions = {
            "title": "Product title/name",
            "price": "Product price in USD",
            "rating": "Customer rating (0-5 scale)",
            "reviews_count": "Number of customer reviews",
            "image_url": "Product image URL",
            "product_url": "Product page URL",
            "availability": "Stock availability status",
            "price_category": "Price category relative to market average",
            "quality_score": "Calculated quality score (0-1)",
            "competitiveness_score": "Market competitiveness score (0-1)",
            "popularity_score": "Popularity based on rating and reviews (0-1)"
        }
        return descriptions.get(field_name, f"Field: {field_name}")


class CSVExporter:
    """
    Advanced CSV exporter for AI/ML model training.
    Supports multiple formats and compression options.
    """
    
    def __init__(self):
        """Initialize CSV exporter"""
        self.supported_formats = ["csv", "json", "parquet", "excel"]
    
    def export_data(
        self,
        data: Dict[str, Any],
        config: CSVExportConfig = None
    ) -> Dict[str, Any]:
        """
        Export processed data to various formats.
        
        Args:
            data: Processed data from DataProcessor
            config: Export configuration
            
        Returns:
            Export results with file contents and metadata
        """
        if config is None:
            config = CSVExportConfig()
        
        try:
            products = data.get("products", [])
            metadata = data.get("metadata", {})
            quality_metrics = data.get("quality_metrics", {})
            
            # Prepare export data
            export_data = self._prepare_export_data(products, metadata, quality_metrics, config)
            
            # Generate files based on format
            files = {}
            
            if config.export_format == "csv":
                files["data.csv"] = self._export_csv(export_data)
            elif config.export_format == "json":
                files["data.json"] = self._export_json(export_data)
            elif config.export_format == "parquet":
                files["data.parquet"] = self._export_parquet(export_data)
            elif config.export_format == "excel":
                files["data.xlsx"] = self._export_excel(export_data)
            
            # Add metadata file
            if config.include_metadata:
                files["metadata.json"] = json.dumps({
                    "export_info": {
                        "exported_at": datetime.now().isoformat(),
                        "format": config.export_format,
                        "total_records": len(products),
                        "data_quality": quality_metrics
                    },
                    "source_metadata": metadata,
                    "schema": data.get("schema", {})
                }, indent=2)
            
            # Apply compression if requested
            if config.compression:
                files = self._compress_files(files, config.compression)
            
            return {
                "files": files,
                "export_metadata": {
                    "format": config.export_format,
                    "compression": config.compression,
                    "total_records": len(products),
                    "file_count": len(files),
                    "export_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def _prepare_export_data(
        self,
        products: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        config: CSVExportConfig
    ) -> List[Dict[str, Any]]:
        """Prepare data for export based on configuration"""
        export_data = products.copy()
        
        if config.include_metadata:
            # Add metadata fields to each record
            for product in export_data:
                product["_source_platform"] = metadata.get("platform", "")
                product["_source_query"] = metadata.get("query", "")
                product["_scraped_at"] = metadata.get("scraped_at", "")
        
        if config.include_quality_metrics:
            # Add quality score to each record
            overall_quality = quality_metrics.get("overall_score", 0)
            for product in export_data:
                product["_data_quality_score"] = overall_quality
        
        return export_data
    
    def _export_csv(self, data: List[Dict[str, Any]]) -> str:
        """Export to CSV format"""
        if not data:
            return ""
        
        output = StringIO()
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    
    def _export_json(self, data: List[Dict[str, Any]]) -> str:
        """Export to JSON format"""
        return json.dumps(data, indent=2, default=str)
    
    def _export_parquet(self, data: List[Dict[str, Any]]) -> bytes:
        """Export to Parquet format"""
        df = pd.DataFrame(data)
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()
    
    def _export_excel(self, data: List[Dict[str, Any]]) -> bytes:
        """Export to Excel format"""
        df = pd.DataFrame(data)
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        return buffer.getvalue()
    
    def _compress_files(self, files: Dict[str, Union[str, bytes]], compression: str) -> Dict[str, bytes]:
        """Compress files based on compression type"""
        if compression == "zip":
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for filename, content in files.items():
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                    zip_file.writestr(filename, content)
            
            return {"export_data.zip": buffer.getvalue()}
        
        # Add other compression methods as needed
        return files
