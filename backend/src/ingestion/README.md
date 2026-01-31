# Multi-Media Data Ingestion System

This module provides a complete data ingestion pipeline for fetching and processing media data from multiple sources (MangaDex, TMDB) and merging them into a unified dataset.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Ingestion Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │   MangaDex   │      │     TMDB     │                     │
│  │    Client    │      │    Client    │                     │
│  └──────┬───────┘      └──────┬───────┘                     │
│         │                     │                              │
│         │   Rate Limiting     │                              │
│         │   Error Handling    │                              │
│         │                     │                              │
│  ┌──────▼───────┐      ┌──────▼───────┐                     │
│  │    Manga     │      │    TMDB      │                     │
│  │  Processor   │      │  Processor   │                     │
│  └──────┬───────┘      └──────┬───────┘                     │
│         │                     │                              │
│         └──────────┬──────────┘                              │
│                    │                                         │
│             ┌──────▼───────┐                                 │
│             │     Data     │                                 │
│             │    Merger    │                                 │
│             └──────┬───────┘                                 │
│                    │                                         │
│             ┌──────▼───────┐                                 │
│             │   Unified    │                                 │
│             │   Dataset    │                                 │
│             └──────────────┘                                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-Source Support**: Fetch data from MangaDex (manga) and TMDB (movies/TV)
- **Rate Limiting**: Automatic rate limiting to respect API limits
- **Error Handling**: Robust retry logic with exponential backoff
- **Checkpointing**: Resume interrupted ingestion from last checkpoint
- **Data Validation**: Automatic validation and cleaning of data
- **Deduplication**: Fuzzy matching to remove duplicate entries
- **Progress Tracking**: Real-time progress bars and logging
- **Parallel Processing**: Async/await for efficient data fetching

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your TMDB_API_KEY
```

## Usage

### 1. Ingest Manga from MangaDex

```bash
# Fetch 5000 manga
python -m src.ingestion.orchestrator ingest \
  --sources mangadex \
  --limit 5000 \
  --output data/processed

# Resume from checkpoint
python -m src.ingestion.orchestrator ingest \
  --sources mangadex \
  --limit 5000 \
  --resume
```

### 2. Ingest Movies/TV from TMDB

```bash
# Fetch movies and TV shows
python -m src.ingestion.orchestrator ingest \
  --sources tmdb \
  --limit 5000 \
  --tmdb-api-key YOUR_API_KEY \
  --output data/processed

# Or use environment variable
export TMDB_API_KEY=your_key_here
python -m src.ingestion.orchestrator ingest \
  --sources tmdb \
  --limit 5000
```

### 3. Ingest from Multiple Sources

```bash
# Fetch from both MangaDex and TMDB
python -m src.ingestion.orchestrator ingest \
  --sources mangadex,tmdb \
  --limit 5000 \
  --tmdb-api-key YOUR_API_KEY
```

### 4. Merge Datasets

```bash
# Merge multiple parquet files
python -m src.ingestion.merger merge \
  --inputs data/processed/manga_final.parquet,data/processed/tmdb_final.parquet \
  --output data/processed/multi_media_v1.parquet \
  --deduplicate \
  --similarity-threshold 90
```

## Configuration

### Environment Variables

```bash
# TMDB API Key (required for TMDB ingestion)
TMDB_API_KEY=your_tmdb_api_key_here

# Rate Limits (optional, defaults shown)
MANGADEX_RATE_LIMIT=5  # requests per second
TMDB_RATE_LIMIT=40     # requests per 10 seconds
```

### Rate Limits

- **MangaDex**: 5 requests/second (no authentication required)
- **TMDB**: 40 requests/10 seconds (requires free API key)

## Data Flow

### 1. Fetching

```python
# MangaDex Client
client = MangaDexClient(rate_limit=5)
await client.initialize()

# Fetch popular manga
manga_data = await client.get_popular(limit=100, offset=0)

# Normalize to MediaBase format
normalized = [client.normalize_to_media_base(item) for item in manga_data]
```

### 2. Processing

```python
# Manga Processor
processor = MangaProcessor()

# Process batch
df = processor.process_batch(normalized)

# Validate data
df = processor.validate_data(df)
```

### 3. Merging

```python
# Data Merger
merger = DataMerger(deduplicate=True, similarity_threshold=90)

# Merge datasets
merged_df = merger.merge_datasets(
    input_files=['manga.parquet', 'tmdb.parquet'],
    output_file='multi_media.parquet'
)
```

## Output Format

The final dataset is saved as a Parquet file with the following schema:

```python
{
    "media_id": str,           # Prefixed ID: "anime-123", "manga-456", "movie-789"
    "title": str,              # Media title
    "synopsis": str,           # Description/overview
    "main_picture": str,       # URL to poster/cover image
    "score": float,            # Rating (0-10)
    "genres": List[str],       # List of genres
    "media_type": str,         # "anime", "manga", "movie", "tv"
    "sub_type": str,           # "Shounen", "K-Drama", "Movie", etc.
    "status": str,             # "ongoing", "completed", etc.
    "release_date": str,       # Release date
    "metadata": Dict,          # Additional source-specific data
}
```

## Checkpointing

The system automatically saves checkpoints every 100 items:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "source": "mangadex",
  "offset": 2500,
  "errors": 12,
  "progress": {
    "movie": 1500,
    "tv": 1000
  }
}
```

To resume from a checkpoint:

```bash
python -m src.ingestion.orchestrator ingest --sources mangadex --resume
```

## Error Handling

The system includes multiple layers of error handling:

1. **Rate Limiting**: Automatic backoff when rate limits are hit
2. **Retry Logic**: Exponential backoff for failed requests (max 3 attempts)
3. **Checkpoint System**: Resume from last successful batch
4. **Error Logging**: Detailed logs of all errors
5. **Graceful Degradation**: Continue processing even if some items fail

## Data Quality

### Validation Rules

**Manga:**

- Required: `media_id`, `title`, `media_type`
- Must have either synopsis or genres
- Filters out R18+ content by default

**Movies/TV:**

- Required: `media_id`, `title`, `media_type`, `synopsis`
- Minimum vote count: 50
- Minimum score: 5.0

### Deduplication

The merger uses fuzzy string matching to detect duplicates:

```python
# Jaccard similarity for title matching
similarity = fuzz.ratio(title1.lower(), title2.lower())

# Threshold: 90% similarity (configurable)
if similarity >= 90:
    # Keep entry with higher score
    keep_higher_scored_entry()
```

## Performance

### Benchmarks

| Source    | Items | Time    | Rate       |
|-----------|-------|---------|------------|
| MangaDex  | 5,000 | ~20 min | 4.2 req/s  |
| TMDB      | 5,000 | ~15 min | 5.5 req/s  |
| Merge     | 10,000| ~2 min  | -          |

### Optimization Tips

1. **Parallel Sources**: Run MangaDex and TMDB ingestion in parallel
2. **Batch Size**: Adjust batch size based on memory (default: 100)
3. **Rate Limits**: Increase if you have premium API access
4. **Checkpoints**: Use checkpoints for large datasets (>10K items)

## Troubleshooting

### Common Issues

**1. Rate Limit Errors**

```bash
# Reduce rate limit
export MANGADEX_RATE_LIMIT=3
export TMDB_RATE_LIMIT=30
```

**2. Missing API Key**

```bash
# Set TMDB API key
export TMDB_API_KEY=your_key_here
```

**3. Memory Issues**

```bash
# Reduce batch size
# Edit orchestrator.py: batch_size=50
```

**4. Checkpoint Corruption**

```bash
# Delete checkpoint and restart
rm data/checkpoints/manga_checkpoint.json
```

## API Clients

### MangaDex Client

```python
from src.media_clients.mangadex_client import MangaDexClient

async with MangaDexClient(rate_limit=5) as client:
    # Search manga
    results = await client.search("One Piece", limit=20)
    
    # Get by ID
    manga = await client.get_by_id("manga-id-here")
    
    # Get popular
    popular = await client.get_popular(limit=100)
    
    # Normalize data
    normalized = client.normalize_to_media_base(manga)
```

### TMDB Client

```python
from src.media_clients.tmdb_client import TMDBClient

async with TMDBClient(api_key="your-key", rate_limit=40) as client:
    # Search movies
    movies = await client.search("Inception", media_type="movie")
    
    # Get by ID
    movie = await client.get_by_id("550", media_type="movie")
    
    # Get popular
    popular = await client.get_popular(media_type="tv")
    
    # Get Asian dramas
    kdramas = await client.get_asian_dramas("KR", limit=100)
    
    # Normalize data
    normalized = client.normalize_to_media_base(movie, media_type="movie")
```

## Next Steps

After ingestion, you can:

1. **Generate Embeddings**: Use the embedding model to create vector representations
2. **Index in Vector DB**: Add to Qdrant/Milvus for semantic search
3. **Update Backend**: Configure backend to use multi-media dataset
4. **Test API**: Test new endpoints for manga, movies, TV search

## Contributing

To add a new data source:

1. Create a new client in `src/media_clients/`
2. Inherit from `MediaClient` base class
3. Implement required methods: `search`, `get_by_id`, `get_popular`, `normalize_to_media_base`
4. Create a processor in `src/data_processing/`
5. Update orchestrator to include new source

## License

MIT License - See LICENSE file for details
