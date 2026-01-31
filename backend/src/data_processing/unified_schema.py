"""Unified schema for all media types across different sources."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date
from enum import Enum


class MediaType(str, Enum):
    """Supported media types."""
    ANIME = "anime"
    MANGA = "manga"
    MOVIE = "movie"
    TV = "tv"
    NOVEL = "novel"


class MediaStatus(str, Enum):
    """Media publication/airing status."""
    ONGOING = "ongoing"
    COMPLETED = "completed"
    UPCOMING = "upcoming"
    HIATUS = "hiatus"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class MediaBase(BaseModel):
    """Unified schema for all media types.
    
    This is the core data model used across all sources (TMDB, MangaDex, MAL).
    All media items are normalized to this format for consistent processing.
    """
    
    # Core identifiers
    media_id: str = Field(..., description="Format: '{type}-{source_id}'")
    title: str = Field(..., description="Primary display title")
    
    # Content
    synopsis: Optional[str] = Field(None, description="Plot summary or description")
    main_picture: Optional[str] = Field(None, description="URL to primary image/poster")
    
    # Ratings and metadata
    score: Optional[float] = Field(None, ge=0, le=10, description="Normalized score 0-10")
    genres: List[str] = Field(default_factory=list, description="Normalized genre tags")
    
    # Classification
    media_type: str = Field(..., description="anime, manga, movie, tv, novel")
    sub_type: Optional[str] = Field(None, description="Shounen, K-Drama, Movie, etc.")
    status: Optional[str] = Field(None, description="ongoing, completed, upcoming")
    
    # Release information
    release_date: Optional[date] = Field(None, description="First release/airing date")
    
    # Extended metadata (source-specific)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific fields")
    
    # Source tracking
    source: str = Field(..., description="tmdb, mangadex, mal")
    source_id: str = Field(..., description="Original ID from source")
    last_updated: date = Field(default_factory=date.today, description="Last update timestamp")
    
    # Schema versioning
    schema_version: str = Field(default="1.0", description="Schema version for migrations")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            date: lambda v: v.isoformat() if v else None
        }


class AnimeExtension(BaseModel):
    """Anime-specific fields extending MediaBase.
    
    Used for anime from MAL and other anime sources.
    """
    
    episodes: Optional[int] = Field(None, ge=0, description="Total episode count")
    studios: List[str] = Field(default_factory=list, description="Production studios")
    broadcast_day: Optional[str] = Field(None, description="Day of week for airing")
    season: Optional[str] = Field(None, description="Winter, Spring, Summer, Fall")
    year: Optional[int] = Field(None, description="Release year")
    
    # Additional anime metadata
    airing_status: Optional[str] = Field(None, description="Currently airing status")
    rating: Optional[str] = Field(None, description="Content rating (PG, R, etc.)")
    duration_minutes: Optional[int] = Field(None, description="Episode duration")
    source_material: Optional[str] = Field(None, description="Manga, Light Novel, Original, etc.")


class MangaExtension(BaseModel):
    """Manga-specific fields extending MediaBase.
    
    Used for manga from MangaDex and other manga sources.
    """
    
    chapters: Optional[int] = Field(None, ge=0, description="Total chapters")
    volumes: Optional[int] = Field(None, ge=0, description="Total volumes")
    authors: List[str] = Field(default_factory=list, description="Authors and artists")
    serialization: Optional[str] = Field(None, description="Serialized in magazine")
    demographic: Optional[str] = Field(None, description="Shounen, Shoujo, Seinen, Josei")
    
    # Additional manga metadata
    content_rating: Optional[str] = Field(None, description="safe, suggestive, erotica, pornographic")
    original_language: Optional[str] = Field(None, description="ja, ko, zh for manga/manhwa/manhua")
    publication_status: Optional[str] = Field(None, description="ongoing, completed, hiatus, cancelled")
    last_chapter: Optional[str] = Field(None, description="Latest chapter number/title")
    follows_count: Optional[int] = Field(None, description="MangaDex follows count")


class MovieExtension(BaseModel):
    """Movie-specific fields extending MediaBase.
    
    Used for movies from TMDB.
    """
    
    runtime_minutes: Optional[int] = Field(None, ge=0, description="Film duration")
    budget: Optional[int] = Field(None, ge=0, description="Production budget in USD")
    revenue: Optional[int] = Field(None, ge=0, description="Box office revenue")
    cast: List[str] = Field(default_factory=list, description="Main cast members")
    directors: List[str] = Field(default_factory=list, description="Film directors")
    production_companies: List[str] = Field(default_factory=list, description="Production studios")
    
    # Additional movie metadata
    certification: Optional[str] = Field(None, description="MPAA/content rating")
    release_date_theater: Optional[date] = Field(None, description="Theatrical release")
    release_date_digital: Optional[date] = Field(None, description="Digital/streaming release")


class TVExtension(BaseModel):
    """TV show-specific fields extending MediaBase.
    
    Used for TV series and dramas from TMDB.
    """
    
    seasons: Optional[int] = Field(None, ge=0, description="Number of seasons")
    total_episodes: Optional[int] = Field(None, ge=0, description="Total episodes across all seasons")
    episode_runtime: Optional[int] = Field(None, ge=0, description="Average episode duration")
    networks: List[str] = Field(default_factory=list, description="Broadcast networks")
    creators: List[str] = Field(default_factory=list, description="Show creators")
    cast: List[str] = Field(default_factory=list, description="Main cast")
    
    # Additional TV metadata
    origin_country: Optional[str] = Field(None, description="Country of origin (KR, JP, CN, US)")
    original_language: Optional[str] = Field(None, description="Original language code")
    in_production: Optional[bool] = Field(None, description="Currently in production")
    last_air_date: Optional[date] = Field(None, description="Final episode date")
    next_episode_date: Optional[date] = Field(None, description="Next episode air date")


class NovelExtension(BaseModel):
    """Light novel/web novel-specific fields extending MediaBase."""
    
    volumes: Optional[int] = Field(None, ge=0, description="Total volumes")
    chapters: Optional[int] = Field(None, ge=0, description="Total chapters")
    authors: List[str] = Field(default_factory=list, description="Authors")
    publisher: Optional[str] = Field(None, description="Light novel publisher")
    
    # Additional novel metadata
    novel_type: Optional[str] = Field(None, description="Light Novel, Web Novel, etc.")
    serialization_platform: Optional[str] = Field(None, description="Platform (Web, Magazine)")


class MediaSourceInfo(BaseModel):
    """Information about a media source.
    
    Tracks source-specific metadata for auditing and debugging.
    """
    
    source: str = Field(..., description="Source name (tmdb, mangadex, mal)")
    source_id: str = Field(..., description="Original ID from source")
    fetch_date: date = Field(default_factory=date.today)
    raw_data_hash: Optional[str] = Field(None, description="Hash of original raw data")
    api_version: Optional[str] = Field(None, description="API version used")
    data_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Quality score 0-1")


def create_media_id(media_type: str, source: str, source_id: str) -> str:
    """Create a standardized media_id.
    
    Args:
        media_type: Type of media (anime, manga, movie, tv, novel)
        source: Source name (tmdb, mangadex, mal)
        source_id: Original ID from source
        
    Returns:
        Standardized media_id string
    """
    return f"{media_type}-{source}-{source_id}"


def parse_media_id(media_id: str) -> Dict[str, str]:
    """Parse a media_id into components.
    
    Args:
        media_id: Standardized media_id string
        
    Returns:
        Dictionary with media_type, source, and source_id
    """
    parts = media_id.split("-", 2)
    if len(parts) == 3:
        return {
            "media_type": parts[0],
            "source": parts[1],
            "source_id": parts[2]
        }
    elif len(parts) == 2:
        # Legacy format: type-source_id
        return {
            "media_type": parts[0],
            "source": "unknown",
            "source_id": parts[1]
        }
    else:
        return {
            "media_type": "unknown",
            "source": "unknown",
            "source_id": media_id
        }


# Genre normalization mappings
GENRE_SYNONYMS = {
    # Action variants
    "action": ["action", " martial arts", "martial arts"],
    # Adventure variants
    "adventure": ["adventure"],
    # Comedy variants
    "comedy": ["comedy", "parody"],
    # Drama variants
    "drama": ["drama"],
    # Fantasy variants
    "fantasy": ["fantasy", "isekai", "supernatural"],
    # Horror variants
    "horror": ["horror", "psychological horror"],
    # Mystery variants
    "mystery": ["mystery", "detective", "thriller"],
    # Romance variants
    "romance": ["romance", "romantic"],
    # Sci-Fi variants
    "sci-fi": ["sci-fi", "science fiction", "mecha", "space"],
    # Slice of Life variants
    "slice of life": ["slice of life", "iyashikei"],
    # Sports variants
    "sports": ["sports"],
}


def normalize_genre(genre: str) -> str:
    """Normalize a genre name to standard form.
    
    Args:
        genre: Raw genre string
        
    Returns:
        Normalized genre name
    """
    genre_lower = genre.lower().strip()
    
    for standard, variants in GENRE_SYNONYMS.items():
        if genre_lower in variants:
            return standard.title()
    
    # Return title-cased original if no match
    return genre.strip().title()


def normalize_genres(genres: List[str]) -> List[str]:
    """Normalize a list of genre names.
    
    Args:
        genres: List of raw genre strings
        
    Returns:
        List of normalized, deduplicated genre names
    """
    normalized = []
    seen = set()
    
    for genre in genres:
        if not genre:
            continue
        norm = normalize_genre(genre)
        if norm and norm not in seen:
            normalized.append(norm)
            seen.add(norm)
    
    return normalized
