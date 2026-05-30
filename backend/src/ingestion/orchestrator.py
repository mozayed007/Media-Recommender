import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm.asyncio import tqdm
import logging
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.media_clients.mangadex_client import MangaDexClient
from src.media_clients.tmdb_client import TMDBClient
from src.media_clients.mal_client import MALClient
from src.data_processing.manga_processor import MangaProcessor
from src.data_processing.tmdb_processor import TMDBProcessor
from src.data_processing.anime_processor import AnimeProcessor
from src.ingestion.checkpoint_manager import CheckpointManager
from src.ingestion.error_handler import IngestionErrorHandler

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()

class IngestionOrchestrator:
    """Orchestrates data ingestion from multiple sources."""
    
    def __init__(
        self,
        output_dir: str = "data/processed",
        checkpoint_dir: str = "data/checkpoints",
        batch_size: int = 100,
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.batch_size = batch_size
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.checkpoint_manager = CheckpointManager(str(self.checkpoint_dir))
        self.error_handler = IngestionErrorHandler()
    
    async def ingest_mal(
        self,
        client_id: str,
        media_types: List[str] = ['anime', 'manga'],
        limit: int = 5000,
        resume: bool = False,
    ) -> pd.DataFrame:
        """Ingest anime/manga data from MyAnimeList.
        
        Args:
            client_id: MAL API client ID
            media_types: List of media types ('anime', 'manga')
            limit: Maximum number of items per type
            resume: Whether to resume from checkpoint
            
        Returns:
            DataFrame with MAL data
        """
        all_data = []
        
        for media_type in media_types:
            source_key = f"mal_{media_type}"
            
            # Load checkpoint
            start_offset = 0
            if resume:
                checkpoint = self.checkpoint_manager.load_checkpoint(source_key)
                if checkpoint:
                    start_offset = checkpoint.get('offset', 0)
                    self.logger.info(f"Resuming MAL {media_type} from offset {start_offset}")
            
            client = MALClient(client_id=client_id, rate_limit=10)
            processor = MangaProcessor() if media_type == 'manga' else AnimeProcessor()
            
            try:
                await client.initialize()
                
                console.print(f"[bold green]Fetching {media_type} from MAL...[/bold green]")
                
                offset = start_offset
                errors = 0
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"Fetching MAL {media_type} ({offset}/{limit})...",
                        total=limit
                    )
                    
                    while offset < limit and errors < 10:
                        try:
                            batch_size = min(self.batch_size, limit - offset)
                            
                            # Fetch using ranking endpoint for bulk data
                            raw_batch = await client.get_ranking(
                                ranking_type="all",
                                media_type=media_type,
                                limit=batch_size,
                                offset=offset
                            )
                            
                            if not raw_batch:
                                self.logger.warning(f"No more data at offset {offset}")
                                break
                            
                            # Normalize data
                            normalized_batch = [
                                client.normalize_to_media_base(item, media_type)
                                for item in raw_batch
                            ]
                            
                            # Process batch
                            df_batch = processor.process_batch(normalized_batch)
                            
                            if not df_batch.empty:
                                all_data.extend(df_batch.to_dict('records'))
                            
                            offset += len(raw_batch)
                            progress.update(
                                task,
                                completed=offset,
                                description=f"Fetching MAL {media_type} ({offset}/{limit})..."
                            )
                            
                            # Save checkpoint periodically
                            if offset % 100 == 0:
                                self.checkpoint_manager.save_checkpoint(
                                    source_key,
                                    {'offset': offset, 'errors': errors}
                                )
                            
                        except Exception as e:
                            self.error_handler.log_error(source_key, e, {'offset': offset})
                            self.logger.error(f"Error at offset {offset}: {e}")
                            errors += 1
                            
                            if self.error_handler.should_retry(e, errors):
                                backoff = self.error_handler.get_backoff_time(errors, e)
                                await asyncio.sleep(backoff)
                            else:
                                offset += self.batch_size
                
                # Final checkpoint save
                if offset > start_offset:
                    self.checkpoint_manager.save_checkpoint(
                        source_key,
                        {'offset': offset, 'errors': errors}
                    )
                    
            finally:
                await client.close()
        
        # Process final data
        if all_data:
            df = pd.DataFrame(all_data)
            df = processor.validate_data(df)
            
            # Save final output
            final_output = self.output_dir / "mal_final.parquet"
            df.to_parquet(final_output, index=False)
            
            console.print(f"[bold green]✓ Ingested {len(df)} MAL entries[/bold green]")
            return df
        
        return pd.DataFrame()
    
    async def ingest_all_sources(
        self,
        sources: List[str],
        config: Dict[str, Any],
        limit_per_source: int = 5000,
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Ingest from multiple sources in parallel or sequentially.
        
        Args:
            sources: List of sources ('mangadex', 'tmdb', 'mal')
            config: Configuration dict with API keys
            limit_per_source: Maximum items per source
            parallel: Whether to run sources in parallel
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        tasks = {}
        
        # Create ingestion tasks
        if 'mangadex' in sources:
            tasks['mangadex'] = self.ingest_manga(
                limit=limit_per_source,
                resume=config.get('resume', False)
            )
        
        if 'tmdb' in sources:
            api_key = config.get('tmdb_api_key')
            if api_key:
                tasks['tmdb'] = self.ingest_tmdb(
                    api_key=api_key,
                    media_types=['movie', 'tv'],
                    limit=limit_per_source,
                    resume=config.get('resume', False)
                )
            else:
                self.logger.warning("TMDB API key not provided, skipping")
        
        if 'mal' in sources:
            client_id = config.get('mal_client_id')
            if client_id:
                tasks['mal'] = self.ingest_mal(
                    client_id=client_id,
                    media_types=['anime', 'manga'],
                    limit=limit_per_source,
                    resume=config.get('resume', False)
                )
            else:
                self.logger.warning("MAL client ID not provided, skipping")
        
        # Execute tasks
        results = {}
        
        if parallel and len(tasks) > 1:
            console.print("[bold cyan]Running ingestion in parallel...[/bold cyan]")
            # Run in parallel
            coroutines = list(tasks.values())
            completed = await asyncio.gather(*coroutines, return_exceptions=True)
            
            for (source, _), result in zip(tasks.items(), completed):
                if isinstance(result, Exception):
                    self.logger.error(f"{source} ingestion failed: {result}")
                    results[source] = pd.DataFrame()
                else:
                    results[source] = result
        else:
            # Run sequentially
            for source, coro in tasks.items():
                console.print(f"[bold cyan]Starting {source}...[/bold cyan]")
                try:
                    results[source] = await coro
                except Exception as e:
                    self.logger.error(f"{source} ingestion failed: {e}")
                    results[source] = pd.DataFrame()
        
        return results
    
    async def ingest_manga(
        self,
        limit: int = 5000,
        resume: bool = False,
    ) -> pd.DataFrame:
        """Ingest manga data from MangaDex.
        
        Args:
            limit: Maximum number of manga to fetch
            resume: Whether to resume from checkpoint
            
        Returns:
            DataFrame with manga data
        """
        checkpoint_file = self.checkpoint_dir / "manga_checkpoint.json"
        output_file = self.output_dir / "manga_partial.parquet"
        
        # Load checkpoint if resuming
        start_offset = 0
        existing_data = []
        
        if resume and checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_offset = checkpoint.get('offset', 0)
                self.logger.info(f"Resuming from offset {start_offset}")
            
            if output_file.exists():
                existing_df = pd.read_parquet(output_file)
                existing_data = existing_df.to_dict('records')
                self.logger.info(f"Loaded {len(existing_data)} existing entries")
        
        client = MangaDexClient(rate_limit=5)
        processor = MangaProcessor()
        
        all_data = existing_data
        errors = 0
        
        try:
            await client.initialize()
            
            console.print(f"[bold green]Fetching manga from MangaDex...[/bold green]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Fetching manga (0/{limit})...", total=limit)
                
                offset = start_offset
                while offset < limit:
                    try:
                        # Fetch batch
                        batch_size = min(self.batch_size, limit - offset)
                        raw_batch = await client.get_popular(limit=batch_size, offset=offset)
                        
                        if not raw_batch:
                            self.logger.warning(f"No more data at offset {offset}")
                            break
                        
                        # Normalize data
                        normalized_batch = [client.normalize_to_media_base(item) for item in raw_batch]
                        
                        # Process batch
                        df_batch = processor.process_batch(normalized_batch)
                        
                        if not df_batch.empty:
                            all_data.extend(df_batch.to_dict('records'))
                        
                        offset += len(raw_batch)
                        progress.update(task, completed=offset, description=f"Fetching manga ({offset}/{limit})...")
                        
                        # Save checkpoint every 100 items
                        if offset % 100 == 0:
                            self._save_checkpoint(checkpoint_file, 'mangadex', offset, errors)
                            self._save_partial_data(all_data, output_file)
                        
                    except Exception as e:
                        self.logger.error(f"Error at offset {offset}: {e}")
                        errors += 1
                        offset += self.batch_size  # Skip this batch
                        
                        if errors > 10:
                            self.logger.error("Too many errors, stopping ingestion")
                            break
            
            # Final processing
            df = pd.DataFrame(all_data)
            df = processor.validate_data(df)
            
            # Save final data
            final_output = self.output_dir / "manga_final.parquet"
            df.to_parquet(final_output, index=False)
            
            console.print(f"[bold green]✓ Ingested {len(df)} manga entries[/bold green]")
            self.logger.info(f"Saved {len(df)} manga to {final_output}")
            
            return df
            
        finally:
            await client.close()
    
    async def ingest_tmdb(
        self,
        api_key: str,
        media_types: List[str] = ['movie', 'tv'],
        limit: int = 5000,
        include_asian_dramas: bool = True,
        resume: bool = False,
    ) -> pd.DataFrame:
        """Ingest movie and TV data from TMDB.
        
        Args:
            api_key: TMDB API key
            media_types: List of media types to fetch ('movie', 'tv')
            limit: Maximum number of items per media type
            include_asian_dramas: Whether to specifically fetch Asian dramas
            resume: Whether to resume from checkpoint
            
        Returns:
            DataFrame with TMDB data
        """
        checkpoint_file = self.checkpoint_dir / "tmdb_checkpoint.json"
        output_file = self.output_dir / "tmdb_partial.parquet"
        
        # Load checkpoint if resuming
        progress_state = {}
        existing_data = []
        
        if resume and checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                progress_state = checkpoint.get('progress', {})
                self.logger.info(f"Resuming TMDB ingestion")
            
            if output_file.exists():
                existing_df = pd.read_parquet(output_file)
                existing_data = existing_df.to_dict('records')
                self.logger.info(f"Loaded {len(existing_data)} existing entries")
        
        client = TMDBClient(api_key=api_key, rate_limit=40)
        processor = TMDBProcessor(min_vote_count=50, min_score=5.0)
        
        all_data = existing_data
        errors = 0
        
        try:
            await client.initialize()
            
            console.print(f"[bold green]Fetching data from TMDB...[/bold green]")
            
            for media_type in media_types:
                offset = progress_state.get(media_type, 0)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"Fetching {media_type} ({offset}/{limit})...",
                        total=limit
                    )
                    
                    while offset < limit:
                        try:
                            batch_size = min(self.batch_size, limit - offset)
                            raw_batch = await client.get_popular(
                                limit=batch_size,
                                offset=offset,
                                media_type=media_type
                            )
                            
                            if not raw_batch:
                                break
                            
                            # Normalize and process
                            normalized_batch = [
                                client.normalize_to_media_base(item, media_type)
                                for item in raw_batch
                            ]
                            
                            df_batch = processor.process_batch(normalized_batch)
                            
                            if not df_batch.empty:
                                all_data.extend(df_batch.to_dict('records'))
                            
                            offset += len(raw_batch)
                            progress.update(
                                task,
                                completed=offset,
                                description=f"Fetching {media_type} ({offset}/{limit})..."
                            )
                            
                            # Save checkpoint
                            if offset % 100 == 0:
                                progress_state[media_type] = offset
                                self._save_checkpoint(
                                    checkpoint_file,
                                    'tmdb',
                                    offset,
                                    errors,
                                    extra={'progress': progress_state}
                                )
                                self._save_partial_data(all_data, output_file)
                            
                        except Exception as e:
                            self.logger.error(f"Error at offset {offset}: {e}")
                            errors += 1
                            offset += self.batch_size
                            
                            if errors > 10:
                                break
            
            # Fetch Asian dramas if requested
            if include_asian_dramas and 'tv' in media_types:
                for country in ['KR', 'JP', 'CN']:
                    offset = progress_state.get(f'drama_{country}', 0)
                    drama_limit = limit // 3
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            f"Fetching {country} dramas ({offset}/{drama_limit})...",
                            total=drama_limit
                        )
                        
                        while offset < drama_limit:
                            try:
                                batch_size = min(self.batch_size, drama_limit - offset)
                                raw_batch = await client.get_asian_dramas(
                                    country_code=country,
                                    limit=batch_size,
                                    offset=offset
                                )
                                
                                if not raw_batch:
                                    break
                                
                                normalized_batch = [
                                    client.normalize_to_media_base(item, 'tv')
                                    for item in raw_batch
                                ]
                                
                                df_batch = processor.process_batch(normalized_batch)
                                
                                if not df_batch.empty:
                                    all_data.extend(df_batch.to_dict('records'))
                                
                                offset += len(raw_batch)
                                progress.update(
                                    task,
                                    completed=offset,
                                    description=f"Fetching {country} dramas ({offset}/{drama_limit})..."
                                )
                                
                            except Exception as e:
                                self.logger.error(f"Error fetching {country} dramas: {e}")
                                errors += 1
                                break
            
            # Final processing
            df = pd.DataFrame(all_data)
            df = processor.validate_data(df)
            
            # Save final data
            final_output = self.output_dir / "tmdb_final.parquet"
            df.to_parquet(final_output, index=False)
            
            console.print(f"[bold green]✓ Ingested {len(df)} TMDB entries[/bold green]")
            self.logger.info(f"Saved {len(df)} TMDB entries to {final_output}")
            
            return df
            
        finally:
            await client.close()
    
    def _save_checkpoint(
        self,
        checkpoint_file: Path,
        source: str,
        offset: int,
        errors: int,
        extra: Optional[Dict] = None
    ):
        """Save checkpoint to file."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'offset': offset,
            'errors': errors,
        }
        
        if extra:
            checkpoint.update(extra)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _save_partial_data(self, data: List[Dict], output_file: Path):
        """Save partial data to parquet."""
        if data:
            df = pd.DataFrame(data)
            df.to_parquet(output_file, index=False)

@app.command()
def ingest(
    sources: str = typer.Option("mangadex,tmdb,mal", help="Comma-separated list of sources"),
    limit: int = typer.Option(5000, help="Maximum items per source"),
    tmdb_api_key: str = typer.Option("", envvar="TMDB_API_KEY", help="TMDB API key"),
    mal_client_id: str = typer.Option("", envvar="MAL_CLIENT_ID", help="MAL client ID"),
    output: str = typer.Option("data/processed", help="Output directory"),
    resume: bool = typer.Option(False, help="Resume from checkpoint"),
    parallel: bool = typer.Option(True, help="Run sources in parallel"),
):
    """Ingest data from multiple sources."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = IngestionOrchestrator(output_dir=output)
    source_list = [s.strip() for s in sources.split(',')]
    
    async def run_ingestion():
        config = {
            'tmdb_api_key': tmdb_api_key,
            'mal_client_id': mal_client_id,
            'resume': resume
        }
        
        results = await orchestrator.ingest_all_sources(
            sources=source_list,
            config=config,
            limit_per_source=limit,
            parallel=parallel
        )
        
        total = sum(len(df) for df in results.values())
        console.print(f"\n[bold green]✓ Ingestion complete! Total: {total} entries[/bold green]")
        
        for source, df in results.items():
            console.print(f"  {source}: {len(df)} entries")
        
        return results
    
    asyncio.run(run_ingestion())

if __name__ == "__main__":
    app()
