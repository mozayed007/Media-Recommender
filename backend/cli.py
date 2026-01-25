import asyncio
import typer
import logging
import os
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from app.services.recommender import RecommenderService
from app.models.anime import RecommendationRequest
from app.core.logging_config import setup_logging

# Setup logging
logger = setup_logging("cli")
console = Console()
app = typer.Typer(help="Media Recommender CLI")

async def get_service(provider: str):
    service = RecommenderService(provider=provider)
    await service.initialize()
    return service

@app.command()
def status(
    provider: str = typer.Option("qdrant", "--provider", "-p", help="Vector DB provider")
):
    """Check the status of the recommendation service."""
    async def _status():
        try:
            console.print(f"[bold blue]Checking status for provider: {provider}...[/bold blue]")
            service = RecommenderService(provider=provider)
            
            try:
                await service.initialize()
            except Exception as e:
                console.print(f"[bold red]Initialization failed: {e}[/bold red]")
                
            table = Table(title="System Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="magenta")
            
            table.add_row(
                "Service", 
                "Ready" if service.is_initialized else "Degraded", 
                f"Provider: {provider}"
            )
            
            data_status = "Loaded" if service.df is not None and not service.df.empty else "Empty"
            table.add_row("Data", data_status, f"{len(service.df) if service.df is not None else 0} records")
            
            if service.vector_db:
                try:
                    count = await service.vector_db.count()
                    table.add_row("Vector DB", "Connected", f"{count} vectors")
                except Exception as e:
                    table.add_row("Vector DB", "Disconnected", f"Error: {str(e)[:50]}...")
            else:
                table.add_row("Vector DB", "Not Initialized", "N/A")
            
            if service.embedding_model:
                model_type = type(service.embedding_model).__name__
                table.add_row("Embedding Model", "Loaded", f"{model_type}")
            else:
                table.add_row("Embedding Model", "Not Loaded", "N/A")
            
            console.print(table)
            
            if not service.is_initialized:
                console.print("\n[bold yellow]Tips for troubleshooting:[/bold yellow]")
                console.print("1. Ensure Docker containers are running or QDRANT_PATH is set for local mode.")
                console.print("2. Check if the data file exists in data/raw/")
                console.print("3. Verify your .env configuration")
                
            logger.info(f"Status check completed for {provider}")
        except Exception as e:
            console.print(f"[bold red]Critical error during status check: {e}[/bold red]")
            logger.error(f"Status check critical failure: {e}", exc_info=True)
            raise typer.Exit(code=1)

    asyncio.run(_status())

@app.command()
def logs(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of last lines to show"),
    component: str = typer.Option("cli", "--component", "-c", help="Component to show logs for (cli, api, recommender)")
):
    """Show the last few lines of the logs."""
    log_file = f"logs/{component}.log"
    if not os.path.exists(log_file):
        log_file = f"backend/logs/{component}.log"
        
    if not os.path.exists(log_file):
        console.print(f"[bold red]Log file not found: {log_file}[/bold red]")
        return
    
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            last_lines = lines[-limit:]
            console.print(f"[bold blue]Last {len(last_lines)} lines of {log_file}:[/bold blue]")
            for line in last_lines:
                console.print(line.strip())
    except Exception as e:
        console.print(f"[bold red]Error reading log file: {e}[/bold red]")

@app.command()
def ingest(
    provider: str = typer.Option("qdrant", "--provider", "-p", help="Vector DB provider"),
    batch_size: int = typer.Option(100, "--batch-size", "-b", help="Batch size for ingestion"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion even if DB is not empty")
):
    """Ingest data into the vector database."""
    async def _ingest():
        try:
            console.print(f"[bold blue]Starting ingestion for provider: {provider}...[/bold blue]")
            service = await get_service(provider)
            
            count = await service.vector_db.count()
            if count > 0 and not force:
                console.print(f"[yellow]Vector DB already has {count} items. Use --force to re-ingest.[/yellow]")
                return
                
            await service.ingest_data(batch_size=batch_size)
            
            console.print("[bold green]Ingestion completed successfully![/bold green]")
            logger.info(f"Ingestion completed for {provider}")
        except Exception as e:
            console.print(f"[bold red]Ingestion failed: {e}[/bold red]")
            logger.error(f"Ingestion failure: {e}", exc_info=True)
            raise typer.Exit(code=1)

    asyncio.run(_ingest())

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (e.g., 'action anime with swords')"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of results to return"),
    provider: str = typer.Option("qdrant", "--provider", "-p", help="Vector DB provider")
):
    """Search for anime using semantic search."""
    async def _search():
        try:
            service = await get_service(provider)
            results = await service.search(query, limit=limit)
            
            table = Table(title=f"Search Results for: '{query}'")
            table.add_column("ID", style="dim")
            table.add_column("Title", style="cyan")
            table.add_column("Similarity", style="green")
            table.add_column("MAL Score", style="magenta")
            
            for res in results:
                # Use similarity_score for the table, and handle None for score
                score_str = f"{res.score:.2f}" if res.score is not None else "N/A"
                table.add_row(str(res.anime_id), res.title, f"{res.similarity_score:.4f}", score_str)
                
            console.print(table)
        except Exception as e:
            console.print(f"[bold red]Search failed: {e}[/bold red]")
            logger.error(f"Search failure: {e}", exc_info=True)
            raise typer.Exit(code=1)

    asyncio.run(_search())

@app.command()
def recommend(
    query: str = typer.Option(None, "--query", "-q", help="Natural language query"),
    anime_id: int = typer.Option(None, "--id", help="Anime ID for similarity search"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of recommendations"),
    provider: str = typer.Option("qdrant", "--provider", "-p", help="Vector DB provider")
):
    """Get recommendations using hybrid search (Query or ID)."""
    async def _recommend():
        try:
            service = await get_service(provider)
            
            if query:
                request = RecommendationRequest(query=query, limit=limit)
            elif anime_id:
                request = RecommendationRequest(anime_id=anime_id, limit=limit)
            else:
                console.print("[bold red]Please provide either --query or --id[/bold red]")
                return
                
            results = await service.get_recommendations(request)
            
            table = Table(title=f"Recommendations")
            table.add_column("Title", style="cyan")
            table.add_column("Similarity", style="green")
            table.add_column("MAL Score", style="magenta")
            
            for rec in results.recommendations:
                score_str = f"{rec.score:.2f}" if rec.score is not None else "N/A"
                sim_str = f"{rec.similarity_score:.4f}"
                table.add_row(rec.title, sim_str, score_str)
            
            console.print(table)
        except Exception as e:
            console.print(f"[bold red]Recommendation failed: {e}[/bold red]")
            logger.error(f"Recommendation failure: {e}", exc_info=True)
            raise typer.Exit(code=1)

    asyncio.run(_recommend())

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    reload: bool = typer.Option(True, help="Enable auto-reload")
):
    """Start the FastAPI backend server."""
    import uvicorn
    console.print(f"[bold green]Starting FastAPI server on {host}:{port}...[/bold green]")
    uvicorn.run("main:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    app()
