from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def process_batch(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a batch of raw data into a DataFrame.
        
        Args:
            raw_data: List of raw data dictionaries
            
        Returns:
            Processed DataFrame
        """
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
        """
        pass
    
    def remove_duplicates(self, df: pd.DataFrame, id_col: str = 'media_id') -> pd.DataFrame:
        """Remove duplicate entries.
        
        Args:
            df: DataFrame
            id_col: Column to use for duplicate detection
            
        Returns:
            DataFrame without duplicates
        """
        initial_count = len(df)
        df = df.drop_duplicates(subset=[id_col], keep='first')
        removed = initial_count - len(df)
        
        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate entries")
        
        return df
    
    def filter_incomplete(self, df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
        """Filter out rows with missing required fields.
        
        Args:
            df: DataFrame
            required_cols: List of required column names
            
        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)
        
        for col in required_cols:
            if col in df.columns:
                df = df[df[col].notna()]
        
        removed = initial_count - len(df)
        if removed > 0:
            self.logger.info(f"Filtered out {removed} incomplete entries")
        
        return df
    
    def normalize_genres(self, genres: List[str]) -> List[str]:
        """Normalize genre names to consistent format.
        
        Args:
            genres: List of genre strings
            
        Returns:
            Normalized genre list
        """
        if not genres:
            return []
        
        # Remove duplicates and normalize case
        normalized = []
        seen = set()
        
        for genre in genres:
            if isinstance(genre, str):
                genre_clean = genre.strip().title()
                if genre_clean and genre_clean not in seen:
                    normalized.append(genre_clean)
                    seen.add(genre_clean)
        
        return normalized
