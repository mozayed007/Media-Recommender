from typing import List, Dict, Any
import pandas as pd
import re
from .base_processor import DataProcessor

class TMDBProcessor(DataProcessor):
    """Processor for movie and TV data from TMDB."""
    
    def __init__(self, min_vote_count: int = 50, min_score: float = 5.0):
        """Initialize TMDB processor.
        
        Args:
            min_vote_count: Minimum vote count for inclusion
            min_score: Minimum score for inclusion
        """
        super().__init__()
        self.min_vote_count = min_vote_count
        self.min_score = min_score
    
    def process_batch(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a batch of TMDB data.
        
        Args:
            raw_data: List of normalized TMDB dictionaries
            
        Returns:
            Processed DataFrame
        """
        if not raw_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # Clean synopses
        if 'synopsis' in df.columns:
            df['synopsis'] = df['synopsis'].apply(self._clean_synopsis)
        
        # Normalize genres
        if 'genres' in df.columns:
            df['genres'] = df['genres'].apply(self.normalize_genres)
        
        # Extract vote count from metadata for filtering
        if 'metadata' in df.columns:
            df['vote_count'] = df['metadata'].apply(lambda x: x.get('vote_count', 0) if isinstance(x, dict) else 0)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate TMDB data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
        """
        required_cols = ['media_id', 'title', 'media_type']
        df = self.filter_incomplete(df, required_cols)
        df = self.remove_duplicates(df, 'media_id')
        
        # Ensure media_type is movie or tv
        if 'media_type' in df.columns:
            df = df[df['media_type'].isin(['movie', 'tv'])]
        
        # Filter by quality metrics
        if 'vote_count' in df.columns:
            df = df[df['vote_count'] >= self.min_vote_count]
        
        if 'score' in df.columns:
            df = df[df['score'] >= self.min_score]
        
        # Filter out entries without synopsis
        if 'synopsis' in df.columns:
            df = df[df['synopsis'].notna() & (df['synopsis'] != '')]
        
        # Drop temporary vote_count column if it was added
        if 'vote_count' in df.columns and 'vote_count' not in required_cols:
            df = df.drop(columns=['vote_count'])
        
        self.logger.info(f"Validated {len(df)} TMDB entries")
        return df
    
    def _clean_synopsis(self, text: Any) -> str:
        """Clean synopsis text.
        
        Args:
            text: Raw synopsis text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common prefixes
        text = re.sub(r'^(Overview:|Synopsis:|Plot:)\s*', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def separate_asian_dramas(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separate Asian dramas from Western content.
        
        Args:
            df: DataFrame with TMDB data
            
        Returns:
            Tuple of (asian_dramas_df, western_content_df)
        """
        if 'metadata' not in df.columns:
            return pd.DataFrame(), df
        
        def is_asian_drama(row):
            if row['media_type'] != 'tv':
                return False
            
            metadata = row.get('metadata', {})
            if not isinstance(metadata, dict):
                return False
            
            origin_countries = metadata.get('origin_country', [])
            asian_countries = {'KR', 'JP', 'CN', 'TW', 'TH'}
            
            return any(country in asian_countries for country in origin_countries)
        
        mask = df.apply(is_asian_drama, axis=1)
        asian_dramas = df[mask].copy()
        western_content = df[~mask].copy()
        
        self.logger.info(f"Separated {len(asian_dramas)} Asian dramas from {len(western_content)} Western content")
        
        return asian_dramas, western_content
