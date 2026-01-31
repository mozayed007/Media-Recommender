"""Data quality checker for media ingestion pipeline.

Provides comprehensive data quality validation and scoring.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Check and validate data quality for media datasets."""
    
    def __init__(
        self,
        min_synopsis_length: int = 50,
        min_score_threshold: float = 0.0,
        require_genres: bool = False,
        require_image: bool = False,
    ):
        """Initialize quality checker.
        
        Args:
            min_synopsis_length: Minimum synopsis length to consider complete
            min_score_threshold: Minimum score threshold (0-10)
            require_genres: Whether genres are required
            require_image: Whether image URL is required
        """
        self.min_synopsis_length = min_synopsis_length
        self.min_score_threshold = min_score_threshold
        self.require_genres = require_genres
        self.require_image = require_image
        self.logger = logging.getLogger(__name__)
    
    def check_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate completeness score for each column.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary mapping column names to completeness percentage
        """
        if df.empty:
            return {}
        
        completeness = {}
        total = len(df)
        
        for col in df.columns:
            non_null = df[col].notna().sum()
            completeness[col] = round(non_null / total * 100, 2) if total > 0 else 0.0
        
        return completeness
    
    def validate_schema(
        self, 
        df: pd.DataFrame, 
        media_type: str = "anime",
        required_fields: Optional[List[str]] = None
    ) -> List[str]:
        """Validate DataFrame against expected schema.
        
        Args:
            df: DataFrame to validate
            media_type: Type of media for validation rules
            required_fields: Override list of required fields
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Default required fields
        if required_fields is None:
            required_fields = ["media_id", "title", "media_type"]
        
        # Check required fields exist
        for field in required_fields:
            if field not in df.columns:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        if "score" in df.columns:
            non_null_scores = df["score"].dropna()
            if len(non_null_scores) > 0:
                try:
                    pd.to_numeric(non_null_scores)
                except (ValueError, TypeError):
                    errors.append("Score column contains non-numeric values")
        
        # Validate score range
        if "score" in df.columns:
            invalid_scores = df[
                df["score"].notna() & 
                ((df["score"] < 0) | (df["score"] > 10))
            ]
            if len(invalid_scores) > 0:
                errors.append(f"Found {len(invalid_scores)} records with invalid score range")
        
        # Check media_type values
        if "media_type" in df.columns:
            valid_types = {"anime", "manga", "movie", "tv", "novel"}
            invalid_types = set(df["media_type"].dropna().unique()) - valid_types
            if invalid_types:
                errors.append(f"Invalid media_type values: {invalid_types}")
        
        # Check genres is list type
        if "genres" in df.columns:
            non_list_genres = df[
                df["genres"].notna() & 
                ~df["genres"].apply(lambda x: isinstance(x, list))
            ]
            if len(non_list_genres) > 0:
                errors.append(f"Found {len(non_list_genres)} records with non-list genres")
        
        return errors
    
    def detect_duplicates(
        self, 
        df: pd.DataFrame, 
        id_col: str = "media_id"
    ) -> pd.DataFrame:
        """Detect duplicate entries in the dataset.
        
        Args:
            df: DataFrame to check
            id_col: Column to check for duplicates
            
        Returns:
            DataFrame of duplicate entries
        """
        if id_col not in df.columns:
            self.logger.warning(f"Column {id_col} not found, cannot detect duplicates")
            return pd.DataFrame()
        
        duplicates = df[df.duplicated(subset=[id_col], keep=False)]
        return duplicates.sort_values(id_col)
    
    def score_quality(self, row: pd.Series) -> float:
        """Calculate quality score for a single record.
        
        Args:
            row: DataFrame row
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        max_score = 5.0  # Total possible points
        
        # Title present (1 point)
        if pd.notna(row.get("title")) and str(row.get("title")).strip():
            score += 1.0
        
        # Synopsis quality (1 point)
        synopsis = row.get("synopsis")
        if pd.notna(synopsis):
            synopsis_len = len(str(synopsis))
            if synopsis_len >= self.min_synopsis_length:
                score += 1.0
            elif synopsis_len > 0:
                score += 0.5
        
        # Score present (1 point)
        if pd.notna(row.get("score")):
            score += 1.0
        
        # Genres present (1 point)
        genres = row.get("genres")
        if isinstance(genres, list) and len(genres) > 0:
            score += 1.0
        
        # Image present (1 point)
        if pd.notna(row.get("main_picture")):
            score += 1.0
        
        return round(score / max_score, 2)
    
    def add_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality score column to DataFrame.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with quality_score column
        """
        if df.empty:
            df["quality_score"] = []
            return df
        
        df["quality_score"] = df.apply(self.score_quality, axis=1)
        return df
    
    def validate_image_urls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate image URLs in the dataset.
        
        Args:
            df: DataFrame with main_picture column
            
        Returns:
            DataFrame with records having invalid image URLs
        """
        if "main_picture" not in df.columns:
            return pd.DataFrame()
        
        invalid_urls = []
        
        for idx, url in df["main_picture"].dropna().items():
            try:
                result = urlparse(str(url))
                if not all([result.scheme, result.netloc]):
                    invalid_urls.append(idx)
            except Exception:
                invalid_urls.append(idx)
        
        return df.loc[invalid_urls] if invalid_urls else pd.DataFrame()
    
    def get_low_quality_records(
        self, 
        df: pd.DataFrame, 
        min_quality_score: float = 0.5
    ) -> pd.DataFrame:
        """Get records with low quality scores.
        
        Args:
            df: DataFrame with quality_score column
            min_quality_score: Minimum acceptable quality score
            
        Returns:
            DataFrame of low quality records
        """
        if "quality_score" not in df.columns:
            df = self.add_quality_scores(df)
        
        return df[df["quality_score"] < min_quality_score]
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {
                "total_records": 0,
                "completeness": {},
                "validation_errors": [],
                "duplicates": 0,
                "avg_quality_score": 0.0,
                "low_quality_records": 0,
            }
        
        # Add quality scores
        df_with_scores = self.add_quality_scores(df.copy())
        
        # Completeness
        completeness = self.check_completeness(df)
        
        # Validation errors
        validation_errors = self.validate_schema(df)
        
        # Duplicates
        duplicates_df = self.detect_duplicates(df)
        
        # Quality stats
        avg_quality = df_with_scores["quality_score"].mean()
        low_quality = len(df_with_scores[df_with_scores["quality_score"] < 0.5])
        
        report = {
            "total_records": len(df),
            "completeness": completeness,
            "validation_errors": validation_errors,
            "duplicates": len(duplicates_df),
            "avg_quality_score": round(avg_quality, 2),
            "low_quality_records": low_quality,
            "quality_distribution": df_with_scores["quality_score"].value_counts().to_dict(),
            "generated_at": datetime.now().isoformat(),
        }
        
        return report
    
    def filter_by_quality(
        self, 
        df: pd.DataFrame, 
        min_quality_score: float = 0.5
    ) -> pd.DataFrame:
        """Filter DataFrame to only include high quality records.
        
        Args:
            df: DataFrame to filter
            min_quality_score: Minimum quality score threshold
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        df = self.add_quality_scores(df)
        filtered = df[df["quality_score"] >= min_quality_score].copy()
        
        # Drop the quality score column if not needed
        if "quality_score" in filtered.columns:
            filtered = filtered.drop(columns=["quality_score"])
        
        self.logger.info(
            f"Quality filter: {len(df)} -> {len(filtered)} records "
            f"(min_score={min_quality_score})"
        )
        
        return filtered.reset_index(drop=True)
