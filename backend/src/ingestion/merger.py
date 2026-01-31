import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from fuzzywuzzy import fuzz
import typer
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()

class DataMerger:
    """Merges data from multiple sources into a unified dataset."""
    
    def __init__(
        self,
        deduplicate: bool = True,
        similarity_threshold: int = 90,
        title_weight: float = 0.7,
        year_weight: float = 0.3,
        priority_order: Optional[List[str]] = None
    ):
        """Initialize data merger.
        
        Args:
            deduplicate: Whether to deduplicate entries
            similarity_threshold: Title similarity threshold for deduplication (0-100)
            title_weight: Weight for title similarity in cross-source matching
            year_weight: Weight for year similarity in cross-source matching
            priority_order: Priority order for sources when resolving conflicts
        """
        self.deduplicate = deduplicate
        self.similarity_threshold = similarity_threshold
        self.title_weight = title_weight
        self.year_weight = year_weight
        self.priority_order = priority_order or ['mal', 'tmdb', 'mangadex']
        self.logger = logging.getLogger(__name__)
    
    def merge_datasets(
        self,
        input_files: List[str],
        output_file: str,
    ) -> pd.DataFrame:
        """Merge multiple parquet files into one.
        
        Args:
            input_files: List of input parquet file paths
            output_file: Output parquet file path
            
        Returns:
            Merged DataFrame
        """
        dfs = []
        
        for file_path in input_files:
            path = Path(file_path)
            if not path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                df = pd.read_parquet(path)
                dfs.append(df)
                self.logger.info(f"Loaded {len(df)} entries from {file_path}")
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
        
        if not dfs:
            raise ValueError("No data loaded from input files")
        
        # Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Combined {len(merged_df)} total entries")
        
        # Deduplicate if requested
        if self.deduplicate:
            merged_df = self._deduplicate(merged_df)
            merged_df = self._deduplicate_cross_source(merged_df)
        
        # Sort by score and popularity
        merged_df = self._sort_by_quality(merged_df)
        
        # Save to output file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(output_path, index=False)
        
        self.logger.info(f"Saved {len(merged_df)} entries to {output_file}")
        
        return merged_df
    
    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries based on title similarity.
        
        Args:
            df: DataFrame to deduplicate
            
        Returns:
            Deduplicated DataFrame
        """
        initial_count = len(df)
        
        # First, remove exact duplicates by media_id
        df = df.drop_duplicates(subset=['media_id'], keep='first')
        
        # Then, check for similar titles across different sources
        if 'title' in df.columns:
            df = self._fuzzy_deduplicate(df)
        
        removed = initial_count - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate entries")
        
        return df
    
    def _deduplicate_cross_source(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Remove cross-source duplicates using advanced matching.
        
        Uses title similarity, year matching, and media type to identify
        duplicates across different sources (e.g., same anime on MAL and TMDB).
        
        Args:
            df: DataFrame to deduplicate
            
        Returns:
            Deduplicated DataFrame
        """
        if len(df) < 2 or 'title' not in df.columns:
            return df
        
        initial_count = len(df)
        
        # Group by media_type for more accurate matching
        media_types = df['media_type'].unique() if 'media_type' in df.columns else ['unknown']
        
        all_kept_indices = []
        
        for media_type in media_types:
            type_df = df[df['media_type'] == media_type] if 'media_type' in df.columns else df
            
            if len(type_df) < 2:
                all_kept_indices.extend(type_df.index.tolist())
                continue
            
            # Get titles and metadata
            titles = type_df['title'].tolist()
            indices = type_df.index.tolist()
            
            # Track which entries to keep
            to_remove = set()
            
            for i in range(len(titles)):
                if indices[i] in to_remove:
                    continue
                
                for j in range(i + 1, len(titles)):
                    if indices[j] in to_remove:
                        continue
                    
                    # Calculate combined similarity score
                    similarity = self._calculate_similarity_score(
                        type_df.loc[indices[i]],
                        type_df.loc[indices[j]]
                    )
                    
                    if similarity >= self.similarity_threshold:
                        # Determine which to keep based on priority
                        keep_idx, remove_idx = self._resolve_conflict(
                            type_df.loc[indices[i]],
                            type_df.loc[indices[j]],
                            indices[i],
                            indices[j]
                        )
                        to_remove.add(remove_idx)
                        self.logger.debug(
                            f"Cross-source duplicate: '{titles[j]}' matches '{titles[i]}' ({similarity:.1f}%)"
                        )
            
            # Keep entries not marked for removal
            for idx in indices:
                if idx not in to_remove:
                    all_kept_indices.append(idx)
        
        # Filter dataframe
        df = df.loc[all_kept_indices].reset_index(drop=True)
        
        removed = initial_count - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} cross-source duplicates")
        
        return df
    
    def _calculate_similarity_score(
        self,
        row1: pd.Series,
        row2: pd.Series
    ) -> float:
        """Calculate combined similarity score between two entries.
        
        Args:
            row1: First entry
            row2: Second entry
            
        Returns:
            Combined similarity score (0-100)
        """
        scores = []
        weights = []
        
        # Title similarity
        title1 = str(row1.get('title', '')).lower()
        title2 = str(row2.get('title', '')).lower()
        
        if title1 and title2:
            # Try exact match first
            if title1 == title2:
                title_sim = 100.0
            else:
                title_sim = fuzz.ratio(title1, title2)
                # Also try token sort ratio for better handling of word order
                token_sim = fuzz.token_sort_ratio(title1, title2)
                title_sim = max(title_sim, token_sim)
            
            scores.append(title_sim)
            weights.append(self.title_weight)
        
        # Year similarity
        year1 = self._extract_year(row1)
        year2 = self._extract_year(row2)
        
        if year1 and year2:
            year_diff = abs(year1 - year2)
            if year_diff == 0:
                year_sim = 100.0
            elif year_diff <= 1:
                year_sim = 80.0
            elif year_diff <= 2:
                year_sim = 50.0
            else:
                year_sim = 0.0
            
            scores.append(year_sim)
            weights.append(self.year_weight)
        
        # Calculate weighted average
        if not scores:
            return 0.0
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return weighted_score
    
    def _extract_year(self, row: pd.Series) -> Optional[int]:
        """Extract year from row data.
        
        Args:
            row: DataFrame row
            
        Returns:
            Year as integer or None
        """
        # Try release_date
        if 'release_date' in row and pd.notna(row['release_date']):
            date_val = row['release_date']
            if isinstance(date_val, str) and len(date_val) >= 4:
                try:
                    return int(date_val[:4])
                except ValueError:
                    pass
        
        # Try year field in metadata
        metadata = row.get('metadata', {})
        if isinstance(metadata, dict):
            year = metadata.get('year')
            if year:
                try:
                    return int(year)
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def _resolve_conflict(
        self,
        row1: pd.Series,
        row2: pd.Series,
        idx1: int,
        idx2: int
    ) -> tuple:
        """Determine which duplicate entry to keep.
        
        Args:
            row1: First entry
            row2: Second entry
            idx1: Index of first entry
            idx2: Index of second entry
            
        Returns:
            Tuple of (keep_index, remove_index)
        """
        # Get sources
        source1 = row1.get('source', 'unknown')
        source2 = row2.get('source', 'unknown')
        
        # If same source, keep the one with higher score
        if source1 == source2:
            score1 = row1.get('score') or 0
            score2 = row2.get('score') or 0
            if score1 >= score2:
                return idx1, idx2
            else:
                return idx2, idx1
        
        # Different sources - use priority order
        priority1 = self._get_source_priority(source1)
        priority2 = self._get_source_priority(source2)
        
        if priority1 < priority2:
            return idx1, idx2
        elif priority2 < priority1:
            return idx2, idx1
        else:
            # Same priority - use score
            score1 = row1.get('score') or 0
            score2 = row2.get('score') or 0
            if score1 >= score2:
                return idx1, idx2
            else:
                return idx2, idx1
    
    def _get_source_priority(self, source: str) -> int:
        """Get priority ranking for a source.
        
        Args:
            source: Source name
            
        Returns:
            Priority value (lower is higher priority)
        """
        try:
            return self.priority_order.index(source.lower())
        except ValueError:
            return len(self.priority_order)
    
    def _fuzzy_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove entries with very similar titles.
        
        Args:
            df: DataFrame to deduplicate
            
        Returns:
            Deduplicated DataFrame
        """
        # Sort by score to keep higher quality entries
        df = df.sort_values('score', ascending=False, na_position='last')
        
        to_remove = set()
        titles = df['title'].tolist()
        
        for i in range(len(titles)):
            if i in to_remove:
                continue
            
            for j in range(i + 1, len(titles)):
                if j in to_remove:
                    continue
                
                # Calculate similarity
                similarity = fuzz.ratio(titles[i].lower(), titles[j].lower())
                
                if similarity >= self.similarity_threshold:
                    # Keep the one with higher score (already sorted)
                    to_remove.add(j)
                    self.logger.debug(f"Marking duplicate: '{titles[j]}' similar to '{titles[i]}' ({similarity}%)")
        
        if to_remove:
            df = df.iloc[[i for i in range(len(df)) if i not in to_remove]]
            self.logger.info(f"Removed {len(to_remove)} fuzzy duplicates")
        
        return df.reset_index(drop=True)
    
    def _sort_by_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort entries by quality metrics.
        
        Args:
            df: DataFrame to sort
            
        Returns:
            Sorted DataFrame
        """
        # Create a quality score combining multiple factors
        df['_quality_score'] = 0.0
        
        # Score contribution (normalized to 0-1)
        if 'score' in df.columns:
            df['_quality_score'] += df['score'].fillna(0) / 10.0
        
        # Popularity from metadata
        if 'metadata' in df.columns:
            def get_popularity(metadata):
                if isinstance(metadata, dict):
                    return metadata.get('popularity', 0) or metadata.get('vote_count', 0)
                return 0
            
            popularity = df['metadata'].apply(get_popularity)
            if popularity.max() > 0:
                df['_quality_score'] += (popularity / popularity.max()) * 0.5
        
        # Sort by quality score
        df = df.sort_values('_quality_score', ascending=False)
        df = df.drop(columns=['_quality_score'])
        
        return df.reset_index(drop=True)
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Get statistics about the merged dataset.
        
        Args:
            df: Merged DataFrame
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_entries': len(df),
            'by_media_type': df['media_type'].value_counts().to_dict() if 'media_type' in df.columns else {},
            'by_sub_type': df['sub_type'].value_counts().to_dict() if 'sub_type' in df.columns else {},
            'with_synopsis': df['synopsis'].notna().sum() if 'synopsis' in df.columns else 0,
            'with_score': df['score'].notna().sum() if 'score' in df.columns else 0,
            'avg_score': df['score'].mean() if 'score' in df.columns else None,
        }
        
        return stats

@app.command()
def merge(
    inputs: str = typer.Option(..., help="Comma-separated list of input parquet files"),
    output: str = typer.Option("data/processed/multi_media_v1.parquet", help="Output file path"),
    deduplicate: bool = typer.Option(True, help="Remove duplicates"),
    similarity_threshold: int = typer.Option(90, help="Title similarity threshold (0-100)"),
    title_weight: float = typer.Option(0.7, help="Weight for title similarity"),
    year_weight: float = typer.Option(0.3, help="Weight for year similarity"),
    priority_order: str = typer.Option("mal,tmdb,mangadex", help="Source priority order (comma-separated)"),
):
    """Merge multiple parquet files into one unified dataset."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    input_files = [f.strip() for f in inputs.split(',')]
    priority_list = [s.strip() for s in priority_order.split(',')]
    
    console.print(f"[bold cyan]Merging {len(input_files)} files...[/bold cyan]")
    
    merger = DataMerger(
        deduplicate=deduplicate,
        similarity_threshold=similarity_threshold,
        title_weight=title_weight,
        year_weight=year_weight,
        priority_order=priority_list
    )
    
    try:
        df = merger.merge_datasets(input_files, output)
        stats = merger.get_statistics(df)
        
        console.print("\n[bold green]✓ Merge complete![/bold green]")
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"  Total entries: {stats['total_entries']}")
        console.print(f"  By media type: {stats['by_media_type']}")
        console.print(f"  With synopsis: {stats['with_synopsis']}")
        console.print(f"  With score: {stats['with_score']}")
        if stats['avg_score']:
            console.print(f"  Average score: {stats['avg_score']:.2f}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise

if __name__ == "__main__":
    app()
