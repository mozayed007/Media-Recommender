"""Anime data processor for MyAnimeList data.

Processes raw MAL anime data into the unified MediaBase format.
"""

from typing import List, Dict, Any
import pandas as pd
import logging
from datetime import datetime

from src.data_processing.base_processor import DataProcessor
from src.data_processing.unified_schema import MediaBase, AnimeExtension, normalize_genres

logger = logging.getLogger(__name__)


class AnimeProcessor(DataProcessor):
    """Processor for anime data from MAL and other anime sources."""
    
    def __init__(
        self,
        min_synopsis_length: int = 50,
        min_score_threshold: float = 0.0,
        require_genres: bool = False
    ):
        """Initialize anime processor.
        
        Args:
            min_synopsis_length: Minimum synopsis length to consider complete
            min_score_threshold: Minimum score threshold (0-10)
            require_genres: Whether to require at least one genre
        """
        super().__init__()
        self.min_synopsis_length = min_synopsis_length
        self.min_score_threshold = min_score_threshold
        self.require_genres = require_genres
    
    def process_batch(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a batch of raw anime data into a DataFrame.
        
        Args:
            raw_data: List of raw anime dictionaries
            
        Returns:
            Processed DataFrame
        """
        if not raw_data:
            return pd.DataFrame()
        
        processed_records = []
        
        for item in raw_data:
            try:
                processed = self._process_single(item)
                if processed:
                    processed_records.append(processed)
            except Exception as e:
                self.logger.warning(f"Failed to process item: {e}")
                continue
        
        if not processed_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(processed_records)
        df = self.validate_data(df)
        
        return df
    
    def _process_single(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single anime record.
        
        Args:
            raw_data: Raw anime data dictionary
            
        Returns:
            Processed dictionary or None if invalid
        """
        # Validate required fields
        media_id = raw_data.get("media_id")
        title = raw_data.get("title")
        
        if not media_id or not title:
            self.logger.debug(f"Skipping item without required fields: media_id={media_id}, title={title}")
            return None
        
        # Normalize genres
        genres = raw_data.get("genres", [])
        if genres:
            genres = normalize_genres(genres)
        
        # Extract anime-specific fields from metadata
        metadata = raw_data.get("metadata", {})
        
        # Build the processed record
        processed = {
            "media_id": media_id,
            "title": title.strip(),
            "synopsis": self._clean_synopsis(raw_data.get("synopsis")),
            "main_picture": raw_data.get("main_picture"),
            "score": self._normalize_score(raw_data.get("score")),
            "genres": genres,
            "media_type": raw_data.get("media_type", "anime"),
            "sub_type": raw_data.get("sub_type"),
            "status": raw_data.get("status", "unknown"),
            "release_date": raw_data.get("release_date"),
            "source": raw_data.get("source", "mal"),
            "source_id": raw_data.get("source_id", ""),
            "last_updated": datetime.now().date(),
            "schema_version": "1.0",
            
            # Anime-specific fields
            "episodes": metadata.get("episodes"),
            "studios": metadata.get("studios", []),
            "broadcast_day": metadata.get("broadcast", {}).get("day_of_the_week"),
            "season": metadata.get("season"),
            "year": metadata.get("year"),
            "duration_minutes": self._normalize_duration(metadata.get("average_episode_duration")),
            "source_material": metadata.get("source_material"),
            "rating": metadata.get("rating"),
            
            # Metadata storage
            "metadata": metadata,
        }
        
        return processed
    
    def _clean_synopsis(self, synopsis: Any) -> Optional[str]:
        """Clean and normalize synopsis text.
        
        Args:
            synopsis: Raw synopsis text
            
        Returns:
            Cleaned synopsis or None
        """
        if not synopsis:
            return None
        
        if not isinstance(synopsis, str):
            return None
        
        # Strip whitespace
        synopsis = synopsis.strip()
        
        # Remove common MAL placeholder text
        placeholders = [
            "No synopsis information has been added to this title.",
            "[Written by MAL Rewrite]",
            "(Source: ",
        ]
        
        for placeholder in placeholders:
            if placeholder in synopsis:
                # Remove the placeholder text
                synopsis = synopsis.replace(placeholder, "").strip()
        
        # Remove empty parentheses that might remain
        synopsis = synopsis.replace("()", "").strip()
        
        return synopsis if synopsis else None
    
    def _normalize_score(self, score: Any) -> Optional[float]:
        """Normalize score to 0-10 scale.
        
        Args:
            score: Raw score value
            
        Returns:
            Normalized score or None
        """
        if score is None:
            return None
        
        try:
            score = float(score)
            # MAL scores are already 0-10
            if 0 <= score <= 10:
                return round(score, 2)
            return None
        except (ValueError, TypeError):
            return None
    
    def _normalize_duration(self, duration_seconds: Any) -> Optional[int]:
        """Convert duration from seconds to minutes.
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            Duration in minutes or None
        """
        if duration_seconds is None:
            return None
        
        try:
            seconds = int(duration_seconds)
            return seconds // 60 if seconds > 0 else None
        except (ValueError, TypeError):
            return None
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean anime data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
        """
        if df.empty:
            return df
        
        initial_count = len(df)
        
        # Remove duplicates by media_id
        df = self.remove_duplicates(df, id_col="media_id")
        
        # Filter by synopsis length if configured
        if self.min_synopsis_length > 0 and "synopsis" in df.columns:
            df = df[
                df["synopsis"].isna() | 
                df["synopsis"].apply(
                    lambda x: len(x) >= self.min_synopsis_length if x else True
                )
            ]
        
        # Filter by score threshold if configured
        if self.min_score_threshold > 0 and "score" in df.columns:
            df = df[
                df["score"].isna() | 
                (df["score"] >= self.min_score_threshold)
            ]
        
        # Filter by genre requirement if configured
        if self.require_genres and "genres" in df.columns:
            df = df[df["genres"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
        
        # Remove entries with empty titles
        df = df[df["title"].notna() & (df["title"].str.strip() != "")]
        
        # Normalize status values
        if "status" in df.columns:
            df["status"] = df["status"].apply(self._normalize_status)
        
        # Ensure lists are properly formatted
        for col in ["genres", "studios"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        
        final_count = len(df)
        if initial_count != final_count:
            self.logger.info(f"Validated data: {initial_count} -> {final_count} entries")
        
        return df.reset_index(drop=True)
    
    def _normalize_status(self, status: Any) -> str:
        """Normalize status value.
        
        Args:
            status: Raw status value
            
        Returns:
            Normalized status string
        """
        if not status:
            return "unknown"
        
        status = str(status).lower().strip()
        
        status_map = {
            "finished_airing": "completed",
            "finished": "completed",
            "completed": "completed",
            "currently_airing": "ongoing",
            "currently_publishing": "ongoing",
            "ongoing": "ongoing",
            "not_yet_aired": "upcoming",
            "not_yet_published": "upcoming",
            "upcoming": "upcoming",
            "on_hiatus": "hiatus",
            "hiatus": "hiatus",
            "cancelled": "cancelled",
            "discontinued": "cancelled",
        }
        
        return status_map.get(status, status)
    
    def enrich_with_anime_extension(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich DataFrame with AnimeExtension fields.
        
        Args:
            df: DataFrame with processed anime data
            
        Returns:
            DataFrame with extension fields
        """
        if df.empty:
            return df
        
        # Create extension fields if they don't exist
        extension_fields = [
            "episodes", "studios", "broadcast_day", 
            "season", "year", "duration_minutes", 
            "source_material", "rating"
        ]
        
        for field in extension_fields:
            if field not in df.columns:
                df[field] = None
        
        return df
    
    def get_quality_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get quality statistics for the processed data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary of quality statistics
        """
        if df.empty:
            return {
                "total_entries": 0,
                "with_synopsis": 0,
                "with_score": 0,
                "with_genres": 0,
                "with_episodes": 0,
                "with_studios": 0,
                "avg_score": None,
            }
        
        stats = {
            "total_entries": len(df),
            "with_synopsis": df["synopsis"].notna().sum() if "synopsis" in df.columns else 0,
            "with_score": df["score"].notna().sum() if "score" in df.columns else 0,
            "with_genres": df["genres"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum() if "genres" in df.columns else 0,
            "with_episodes": df["episodes"].notna().sum() if "episodes" in df.columns else 0,
            "with_studios": df["studios"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum() if "studios" in df.columns else 0,
            "avg_score": df["score"].mean() if "score" in df.columns else None,
            "by_status": df["status"].value_counts().to_dict() if "status" in df.columns else {},
        }
        
        return stats
