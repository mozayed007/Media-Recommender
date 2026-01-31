# Phase 1 Implementation: Multi-Source API Infrastructure & Environment Setup

Immediate implementation plan for establishing the foundation of multi-source media scraping with TMDB, MangaDex, and MAL API integration.

## Scope

This implementation focuses on Phase 1 of the comprehensive plan: setting up API client infrastructure, environment configuration, and data processing pipeline to support multiple media sources (TMDB, MangaDex, MAL) with the newly added API keys.

---

## Step 1: Environment Configuration Enhancement

### 1.1 Update Configuration File

**File**: `backend/app/core/config.py`

**Changes**:

- Add `TMDB_READ_ACCESS_TOKEN` field for TMDB v4 API
- Add `MANGADEX_API_KEY` field for authenticated MangaDex requests
- Add retry policy configurations (max_retries, backoff_factor)
- Add timeout configurations per API source
- Add batch size configurations for ingestion
- Add checkpoint directory configuration
- Add data quality thresholds

**New Fields**:

```python
# API Keys
TMDB_READ_ACCESS_TOKEN: Optional[str] = None
MANGADEX_API_KEY: Optional[str] = None

# Retry & Timeout Configs
API_MAX_RETRIES: int = 3
API_BACKOFF_FACTOR: float = 2.0
API_TIMEOUT: int = 30
TMDB_TIMEOUT: int = 30
MANGADEX_TIMEOUT: int = 30
MAL_TIMEOUT: int = 30

# Ingestion Configs
INGESTION_BATCH_SIZE: int = 100
CHECKPOINT_DIR: str = "backend/data/checkpoints"
PROCESSED_DATA_DIR: str = "backend/data/processed"

# Data Quality
MIN_SYNOPSIS_LENGTH: int = 50
MIN_SCORE_THRESHOLD: float = 0.0
REQUIRE_GENRES: bool = False
```

### 1.2 Update .env.example

**File**: `backend/.env.example`

**Add**:

```bash
# TMDB API (v3 and v4)
TMDB_API_KEY=your_tmdb_api_key_here
TMDB_READ_ACCESS_TOKEN=your_tmdb_read_access_token_here

# MangaDex API
MANGADEX_API_KEY=your_mangadex_api_key_here

# MyAnimeList API
MAL_CLIENT_ID=your_mal_client_id_here

# Retry & Timeout
API_MAX_RETRIES=3
API_BACKOFF_FACTOR=2.0
API_TIMEOUT=30

# Ingestion
INGESTION_BATCH_SIZE=100
```

---

## Step 2: Enhance Existing API Clients

### 2.1 TMDB Client Enhancement

**File**: `backend/src/media_clients/tmdb_client.py`

**Enhancements**:

1. **Add v4 API Support**:
   - Add `read_access_token` parameter to `__init__`
   - Create `_make_v4_request` method using Bearer token authentication
   - Add `use_v4` flag to switch between v3 and v4 APIs

2. **Advanced Filtering**:
   - Add `get_by_filters` method with parameters:
     - `language`: Filter by original language
     - `region`: Filter by production country
     - `certification`: Filter by content rating
     - `min_vote_count`: Minimum votes threshold
     - `min_score`: Minimum rating threshold

3. **Enhanced Metadata**:
   - Add `get_detailed` method to fetch full details including:
     - Cast and crew
     - Production companies
     - Keywords
     - Streaming availability (via watch providers endpoint)
   - Add `get_season_details` for TV shows

4. **Genre Mapping**:
   - Create `TMDB_GENRE_MAP` dictionary for ID to name mapping
   - Add `_map_genre_ids` method to convert genre IDs to names

### 2.2 MangaDex Client Enhancement

**File**: `backend/src/media_clients/mangadex_client.py`

**Enhancements**:

1. **Authenticated Requests**:
   - Add `api_key` parameter to `__init__`
   - Update `_make_request` to include API key in headers if provided
   - Add session token management for authenticated endpoints

2. **Chapter Statistics**:
   - Add `get_statistics` method to fetch:
     - Total chapters
     - Latest chapter
     - Reading statistics
     - Follows count

3. **Enhanced Filtering**:
   - Add `publication_demographic` filter (shounen, seinen, shoujo, josei)
   - Add `original_language` filter for manhwa/manhua detection
   - Add `content_rating` filter with SFW option

4. **Author/Artist Metadata**:
   - Enhance `_extract_authors` to include:
     - Author biography
     - Other works
     - Social media links

---

## Step 3: Create MAL API Client

### 3.1 New MAL Client

**File**: `backend/src/media_clients/mal_client.py`

**Implementation**:

```python
class MALClient(MediaClient):
    """Client for MyAnimeList API v2"""
    
    def __init__(self, client_id: str, rate_limit: int = 10):
        super().__init__(
            api_key=client_id,
            base_url="https://api.myanimelist.net/v2"
        )
        self.rate_limiter = RateLimiter(rate=rate_limit, per=1.0)
    
    # Methods to implement:
    async def search(query, media_type, limit, offset)
    async def get_by_id(media_id, media_type)
    async def get_ranking(ranking_type, media_type, limit, offset)
    async def get_seasonal(year, season, limit, offset)
    async def get_suggestions(limit, offset)
    def normalize_to_media_base(raw_data, media_type)
```

**Features**:

- Support both anime and manga endpoints
- Implement ranking types: all, airing, upcoming, tv, movie, ova, special, bypopularity, favorite
- Add seasonal anime fetching (winter, spring, summer, fall)
- Implement user suggestions endpoint (requires OAuth)
- Full field expansion for detailed metadata

---

## Step 4: Data Processing Pipeline

### 4.1 Unified Schema Definition

**File**: `backend/src/data_processing/unified_schema.py`

**Create MediaBase Schema**:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date

class MediaBase(BaseModel):
    """Unified schema for all media types"""
    media_id: str  # Format: "{type}-{source_id}"
    title: str
    synopsis: Optional[str] = None
    main_picture: Optional[str] = None
    score: Optional[float] = None
    genres: List[str] = []
    media_type: str  # anime, manga, movie, tv, novel
    sub_type: Optional[str] = None  # Shounen, K-Drama, Movie, etc.
    status: Optional[str] = None  # ongoing, completed, upcoming
    release_date: Optional[date] = None
    metadata: Dict[str, Any] = {}
    
    # Source tracking
    source: str  # tmdb, mangadex, mal
    source_id: str
    last_updated: date = Field(default_factory=date.today)

class AnimeExtension(BaseModel):
    """Anime-specific fields"""
    episodes: Optional[int] = None
    studios: List[str] = []
    broadcast_day: Optional[str] = None
    season: Optional[str] = None
    year: Optional[int] = None

class MangaExtension(BaseModel):
    """Manga-specific fields"""
    chapters: Optional[int] = None
    volumes: Optional[int] = None
    authors: List[str] = []
    serialization: Optional[str] = None
    demographic: Optional[str] = None

# Similar extensions for Movie, TV, Novel
```

### 4.2 Create Anime Processor

**File**: `backend/src/data_processing/anime_processor.py`

**Implementation**:

- Inherit from `DataProcessor` base class
- Process MAL anime data to MediaBase format
- Handle episode count normalization
- Process airing status (currently_airing, finished_airing, not_yet_aired)
- Extract studio and producer information
- Validate required fields: media_id, title, media_type

### 4.3 Enhance Existing Processors

**Files**:

- `backend/src/data_processing/manga_processor.py`
- `backend/src/data_processing/tmdb_processor.py`

**Enhancements**:

- Update to use unified MediaBase schema
- Add schema version tracking
- Implement data enrichment (fetch missing fields from alternative sources)
- Add validation for required fields
- Implement genre normalization across sources

### 4.4 Create Quality Checker

**File**: `backend/src/data_processing/quality_checker.py`

**Features**:

```python
class DataQualityChecker:
    def check_completeness(df: pd.DataFrame) -> Dict[str, float]
    def validate_schema(df: pd.DataFrame, media_type: str) -> List[str]
    def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame
    def score_quality(row: pd.Series) -> float
    def enrich_missing_fields(df: pd.DataFrame) -> pd.DataFrame
```

**Quality Metrics**:

- Completeness score (% of non-null fields)
- Synopsis length validation
- Genre presence check
- Score range validation (0-10)
- Image URL validation

---

## Step 5: Enhanced Ingestion Orchestrator

### 5.1 Update Orchestrator

**File**: `backend/src/ingestion/orchestrator.py`

**Enhancements**:

1. **Add MAL Ingestion**:

```python
async def ingest_mal(
    self,
    client_id: str,
    media_types: List[str] = ['anime', 'manga'],
    limit: int = 5000,
    resume: bool = False,
) -> pd.DataFrame
```

1. **Parallel Source Ingestion**:

```python
async def ingest_all_sources(
    self,
    sources: List[str],
    limit_per_source: int = 5000,
    parallel: bool = True
) -> Dict[str, pd.DataFrame]
```

1. **Incremental Updates**:

```python
async def ingest_incremental(
    self,
    source: str,
    since_date: date,
    limit: int = 1000
) -> pd.DataFrame
```

### 5.2 Create Checkpoint Manager

**File**: `backend/src/ingestion/checkpoint_manager.py`

**Features**:

```python
class CheckpointManager:
    def save_checkpoint(source: str, state: Dict)
    def load_checkpoint(source: str) -> Optional[Dict]
    def clear_checkpoint(source: str)
    def list_checkpoints() -> List[str]
    def get_checkpoint_age(source: str) -> timedelta
```

**Checkpoint Structure**:

```json
{
  "source": "tmdb",
  "media_type": "movie",
  "timestamp": "2024-01-15T10:30:00",
  "offset": 2500,
  "total_fetched": 2500,
  "errors": 12,
  "last_id": "movie-12345",
  "version": "1.0"
}
```

### 5.3 Create Error Handler

**File**: `backend/src/ingestion/error_handler.py`

**Features**:

```python
class IngestionErrorHandler:
    def categorize_error(error: Exception) -> str  # transient, permanent, rate_limit
    def should_retry(error: Exception, attempt: int) -> bool
    def get_backoff_time(attempt: int) -> float
    def log_error(source: str, error: Exception, context: Dict)
    def get_error_stats(source: str) -> Dict
```

**Error Categories**:

- Transient: Network errors, timeouts
- Rate Limit: 429 responses
- Permanent: 404, 401, invalid data
- Unknown: Unexpected errors

---

## Step 6: Data Merger Enhancement

### 6.1 Update Merger

**File**: `backend/src/ingestion/merger.py`

**Enhancements**:

1. **Cross-Source Deduplication**:

```python
def deduplicate_cross_source(
    datasets: List[pd.DataFrame],
    similarity_threshold: int = 90,
    title_weight: float = 0.7,
    year_weight: float = 0.3
) -> pd.DataFrame
```

1. **Conflict Resolution**:

```python
def resolve_conflicts(
    duplicate_group: pd.DataFrame,
    priority_order: List[str] = ['mal', 'tmdb', 'mangadex']
) -> pd.Series
```

1. **Multi-Language Title Matching**:

```python
def match_titles_multilang(
    title1: str,
    title2: str,
    alt_titles1: List[str],
    alt_titles2: List[str]
) -> float
```

---

## Step 7: Testing Infrastructure

### 7.1 API Client Tests

**Directory**: `backend/tests/integration/api_clients/`

**Files**:

- `test_tmdb_client.py`: Test TMDB v3 and v4 APIs
- `test_mangadex_client.py`: Test authenticated and unauthenticated requests
- `test_mal_client.py`: Test all MAL endpoints

**Test Cases**:

- API authentication
- Rate limiting
- Error handling
- Data normalization
- Pagination

### 7.2 Data Processing Tests

**Directory**: `backend/tests/unit/data_processing/`

**Files**:

- `test_unified_schema.py`: Schema validation
- `test_processors.py`: Processor logic
- `test_quality_checker.py`: Quality metrics

### 7.3 Ingestion Tests

**Directory**: `backend/tests/integration/ingestion/`

**Files**:

- `test_orchestrator.py`: End-to-end ingestion
- `test_checkpoint_manager.py`: Checkpoint persistence
- `test_merger.py`: Deduplication logic

---

## Step 8: CLI Tools

### 8.1 Ingestion CLI

**File**: `backend/cli.py` (enhance existing)

**Add Commands**:

```bash
# Ingest from single source
python cli.py ingest --source tmdb --media-type movie --limit 5000

# Ingest from all sources
python cli.py ingest-all --limit 5000 --parallel

# Incremental update
python cli.py update --source mal --since 2024-01-01

# Merge datasets
python cli.py merge --inputs tmdb.parquet,mal.parquet --output merged.parquet

# Quality check
python cli.py quality-check --input merged.parquet --report quality_report.json
```

### 8.2 Data Management CLI

**New Commands**:

```bash
# List checkpoints
python cli.py checkpoints list

# Resume from checkpoint
python cli.py ingest --source tmdb --resume

# Clear checkpoint
python cli.py checkpoints clear --source tmdb

# View ingestion stats
python cli.py stats --source all
```

---

## Implementation Order

### Week 1: Foundation

1. ✅ Update `config.py` with new fields
2. ✅ Update `.env.example`
3. ✅ Create unified schema (`unified_schema.py`)
4. ✅ Create MAL client (`mal_client.py`)
5. ✅ Enhance TMDB client
6. ✅ Enhance MangaDex client

### Week 2: Processing & Quality

1. ✅ Create anime processor
2. ✅ Update manga processor
3. ✅ Update TMDB processor
4. ✅ Create quality checker
5. ✅ Create checkpoint manager
6. ✅ Create error handler

### Week 3: Orchestration & Testing

1. ✅ Update orchestrator with MAL support
2. ✅ Add parallel ingestion
3. ✅ Enhance merger with cross-source dedup
4. ✅ Write API client tests
5. ✅ Write data processing tests
6. ✅ Write ingestion tests

### Week 4: CLI & Documentation

1. ✅ Enhance CLI with new commands
2. ✅ Create usage documentation
3. ✅ Create API integration guide
4. ✅ End-to-end testing
5. ✅ Performance optimization

---

## Success Criteria

### Functional Requirements

- ✅ All three API clients (TMDB, MangaDex, MAL) working with authentication
- ✅ Data normalized to unified MediaBase schema
- ✅ Cross-source deduplication achieving >95% accuracy
- ✅ Checkpoint system allows resume from any point
- ✅ Error handling with automatic retry for transient failures

### Performance Requirements

- ✅ TMDB: 40 req/10s (v3) or higher with v4 token
- ✅ MangaDex: 5 req/s (unauthenticated) or 10 req/s (authenticated)
- ✅ MAL: 10 req/s
- ✅ Ingestion of 10K items completes in <30 minutes per source
- ✅ Merge operation on 50K items completes in <5 minutes

### Quality Requirements

- ✅ Data completeness >90% for core fields (title, synopsis, genres)
- ✅ Duplicate rate <1% after deduplication
- ✅ Schema validation passes for 100% of records
- ✅ Error rate <5% during ingestion

---

## Dependencies

### Python Packages (add to requirements.txt)

```python
# Already installed
aiohttp>=3.9.0
tenacity>=8.0.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.21.0

# May need to add
pydantic>=2.0.0
```

### External Services

- TMDB API account (free tier sufficient)
- MangaDex API (no account needed for basic, account for higher limits)
- MyAnimeList API (free, requires client ID)

### Infrastructure

- Disk space: ~10GB for raw data, ~5GB for processed data
- Memory: 8GB RAM minimum for processing large datasets
- Network: Stable connection for API calls

---

## Risk Mitigation

### API Rate Limits

- **Risk**: Hitting rate limits during bulk ingestion
- **Mitigation**:
  - Implement intelligent rate limiting with token bucket
  - Use TMDB v4 API for higher limits
  - Distribute ingestion across multiple time periods
  - Cache responses aggressively

### Data Quality

- **Risk**: Inconsistent data across sources
- **Mitigation**:
  - Implement strict schema validation
  - Use quality scoring to filter low-quality entries
  - Manual review of edge cases
  - Implement data enrichment from multiple sources

### API Changes

- **Risk**: API endpoints or schemas change
- **Mitigation**:
  - Version all API client code
  - Implement schema validation with clear error messages
  - Monitor API changelog
  - Graceful degradation for missing fields

### Checkpoint Corruption

- **Risk**: Checkpoint files become corrupted
- **Mitigation**:
  - Atomic writes for checkpoint files
  - Checkpoint versioning
  - Backup checkpoints before updates
  - Validation on checkpoint load

---

## Next Steps After Phase 1

Once Phase 1 is complete, proceed to:

- **Phase 2**: Production embedding system optimization
- **Phase 3**: Vector database production setup
- **Phase 4**: Enhanced recommendation engine
- **Phase 5**: API enhancements and caching
- **Phase 6**: Monitoring and observability

This implementation provides the foundation for a robust, multi-source media recommendation system with production-grade data quality and reliability.
