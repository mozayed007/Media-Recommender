from typing import List, Dict, Any
import pandas as pd
import re
from .base_processor import DataProcessor

class MangaProcessor(DataProcessor):
    """Processor for manga data from MangaDex."""
    
    def process_batch(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a batch of manga data.
        
        Args:
            raw_data: List of normalized manga dictionaries
            
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
        
        # Handle missing synopses
        df = self._handle_missing_synopses(df)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate manga data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
        """
        required_cols = ['media_id', 'title', 'media_type']
        df = self.filter_incomplete(df, required_cols)
        df = self.remove_duplicates(df, 'media_id')
        
        # Ensure media_type is correct
        if 'media_type' in df.columns:
            df = df[df['media_type'] == 'manga']
        
        # Filter out entries with no synopsis and no genres
        mask = (df['synopsis'].notna()) | (df['genres'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))
        df = df[mask]
        
        self.logger.info(f"Validated {len(df)} manga entries")
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
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common prefixes
        text = re.sub(r'^(Synopsis:|Description:)\s*', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _handle_missing_synopses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle entries with missing synopses.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with handled missing synopses
        """
        if 'synopsis' not in df.columns:
            return df
        
        # For entries without synopsis, create one from title and genres
        missing_mask = df['synopsis'].isna() | (df['synopsis'] == '')
        
        if missing_mask.any():
            def create_fallback_synopsis(row):
                if pd.isna(row['synopsis']) or row['synopsis'] == '':
                    genres_str = ', '.join(row['genres']) if isinstance(row['genres'], list) and row['genres'] else 'various genres'
                    return f"A {row['sub_type']} manga titled '{row['title']}' featuring {genres_str}."
                return row['synopsis']
            
            df['synopsis'] = df.apply(create_fallback_synopsis, axis=1)
            self.logger.info(f"Created fallback synopses for {missing_mask.sum()} entries")
        
        return df
