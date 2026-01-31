"""MyAnimeList (MAL) API Client.

Implements the MAL API v2 client for fetching anime and manga data.
https://myanimelist.net/apiconfig/references/api/v2
"""

from typing import List, Dict, Any, Optional
import aiohttp
import logging
from src.media_clients.base_client import MediaClient
from src.media_clients.utils import RateLimiter, with_retry, NotFoundError, APIError

logger = logging.getLogger(__name__)

# MAL API field expansions for detailed responses
ANIME_FIELDS = [
    "id", "title", "main_picture", "alternative_titles", "start_date", "end_date",
    "synopsis", "mean", "rank", "popularity", "num_list_users", "num_scoring_users",
    "nsfw", "genres", "created_at", "updated_at", "media_type", "status",
    "num_episodes", "start_season", "broadcast", "source", "average_episode_duration",
    "rating", "studios", "pictures", "background", "related_anime", "related_manga",
    "recommendations"
]

MANGA_FIELDS = [
    "id", "title", "main_picture", "alternative_titles", "start_date", "end_date",
    "synopsis", "mean", "rank", "popularity", "num_list_users", "num_scoring_users",
    "nsfw", "genres", "created_at", "updated_at", "media_type", "status",
    "num_volumes", "num_chapters", "authors", "pictures", "background",
    "related_anime", "related_manga", "recommendations"
]

# Media type mappings
MAL_MEDIA_TYPES = {
    "anime": ["tv", "movie", "ova", "special", "ona", "music"],
    "manga": ["manga", "novel", "one_shot", "doujinshi", "manhwa", "manhua"]
}

# Ranking types
RANKING_TYPES = [
    "all", "airing", "upcoming", "tv", "movie", "ova", "special",
    "bypopularity", "favorite"
]

# Season mappings
SEASONS = ["winter", "spring", "summer", "fall"]

# Content ratings
CONTENT_RATINGS = {
    "g": "G - All Ages",
    "pg": "PG - Children",
    "pg_13": "PG-13 - Teens 13+",
    "r": "R - 17+",
    "r+": "R+ - Mild Nudity",
    "rx": "Rx - Hentai"
}


class MALClient(MediaClient):
    """Client for MyAnimeList API v2.
    
    Supports both anime and manga endpoints with rate limiting
    and automatic retry logic.
    """
    
    def __init__(self, client_id: str, rate_limit: int = 10):
        """Initialize MAL client.
        
        Args:
            client_id: MAL API client ID
            rate_limit: Requests per second (default: 10, MAL limit)
        """
        super().__init__(
            api_key=client_id,
            base_url="https://api.myanimelist.net/v2"
        )
        self.rate_limiter = RateLimiter(rate=rate_limit, per=1.0)
        self.logger = logging.getLogger(__name__)
    
    @with_retry(max_attempts=3)
    async def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make rate-limited request to MAL API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response
        """
        await self.rate_limiter.acquire()
        
        if not self.session:
            await self.initialize()
        
        url = f"{self.base_url}{endpoint}"
        
        headers = {
            "X-MAL-CLIENT-ID": self.api_key
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 404:
                    raise NotFoundError(f"Resource not found: {url}")
                elif response.status == 429:
                    self.logger.warning("Rate limit exceeded, backing off...")
                    raise APIError("Rate limit exceeded")
                elif response.status == 401:
                    raise APIError("Invalid client ID")
                elif response.status != 200:
                    error_text = await response.text()
                    raise APIError(f"API request failed with status {response.status}: {error_text}")
                
                return await response.json()
        except aiohttp.ClientError as e:
            self.logger.error(f"Request failed: {e}")
            raise APIError(f"Request failed: {e}")
    
    async def search(
        self, 
        query: str, 
        media_type: str = "anime",
        limit: int = 20, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search for anime or manga by title.
        
        Args:
            query: Search query
            media_type: "anime" or "manga"
            limit: Maximum results (max 100)
            offset: Pagination offset
            
        Returns:
            List of media data
        """
        fields = ANIME_FIELDS if media_type == "anime" else MANGA_FIELDS
        
        params = {
            "q": query,
            "limit": min(limit, 100),
            "offset": offset,
            "fields": ",".join(fields)
        }
        
        endpoint = f"/{media_type}"
        response = await self._make_request(endpoint, params)
        
        return response.get("data", [])
    
    async def get_by_id(
        self, 
        media_id: int, 
        media_type: str = "anime"
    ) -> Dict[str, Any]:
        """Get anime or manga details by ID.
        
        Args:
            media_id: MAL ID
            media_type: "anime" or "manga"
            
        Returns:
            Media data
        """
        fields = ANIME_FIELDS if media_type == "anime" else MANGA_FIELDS
        
        params = {
            "fields": ",".join(fields)
        }
        
        endpoint = f"/{media_type}/{media_id}"
        return await self._make_request(endpoint, params)
    
    async def get_ranking(
        self,
        ranking_type: str = "all",
        media_type: str = "anime",
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get ranked list of anime or manga.
        
        Args:
            ranking_type: Type of ranking (all, airing, upcoming, tv, movie, ova,
                        special, bypopularity, favorite)
            media_type: "anime" or "manga"
            limit: Maximum results (max 500)
            offset: Pagination offset
            
        Returns:
            List of ranked media
        """
        if ranking_type not in RANKING_TYPES:
            raise ValueError(f"Invalid ranking_type. Must be one of: {RANKING_TYPES}")
        
        fields = ANIME_FIELDS if media_type == "anime" else MANGA_FIELDS
        
        params = {
            "ranking_type": ranking_type,
            "limit": min(limit, 500),
            "offset": offset,
            "fields": ",".join(fields)
        }
        
        endpoint = f"/{media_type}/ranking"
        response = await self._make_request(endpoint, params)
        
        return response.get("data", [])
    
    async def get_seasonal(
        self,
        year: int,
        season: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get seasonal anime list.
        
        Args:
            year: Year (e.g., 2024)
            season: Season (winter, spring, summer, fall)
            limit: Maximum results (max 500)
            offset: Pagination offset
            
        Returns:
            List of seasonal anime
        """
        if season not in SEASONS:
            raise ValueError(f"Invalid season. Must be one of: {SEASONS}")
        
        params = {
            "limit": min(limit, 500),
            "offset": offset,
            "fields": ",".join(ANIME_FIELDS)
        }
        
        endpoint = f"/anime/season/{year}/{season}"
        response = await self._make_request(endpoint, params)
        
        return response.get("data", [])
    
    async def get_suggestions(
        self,
        media_type: str = "anime",
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get anime/manga suggestions (requires OAuth, may not work with client ID only).
        
        Args:
            media_type: "anime" or "manga"
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of suggested media
        """
        fields = ANIME_FIELDS if media_type == "anime" else MANGA_FIELDS
        
        params = {
            "limit": min(limit, 100),
            "offset": offset,
            "fields": ",".join(fields)
        }
        
        endpoint = f"/{media_type}/suggestions"
        
        try:
            response = await self._make_request(endpoint, params)
            return response.get("data", [])
        except APIError as e:
            self.logger.warning(f"Suggestions endpoint requires OAuth: {e}")
            return []
    
    def _extract_genres(self, node: Dict[str, Any]) -> List[str]:
        """Extract genre names from MAL genre data."""
        genres = []
        for genre_data in node.get("genres", []):
            genre_name = genre_data.get("name")
            if genre_name:
                genres.append(genre_name)
        return genres
    
    def _extract_studios(self, node: Dict[str, Any]) -> List[str]:
        """Extract studio names from MAL studio data."""
        studios = []
        for studio_data in node.get("studios", []):
            studio_name = studio_data.get("name")
            if studio_name:
                studios.append(studio_name)
        return studios
    
    def _extract_authors(self, node: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract author information from MAL author data."""
        authors = []
        for author_data in node.get("authors", []):
            author_info = {
                "name": author_data.get("node", {}).get("first_name", "") + " " + 
                       author_data.get("node", {}).get("last_name", ""),
                "role": author_data.get("role", "")
            }
            if author_info["name"].strip():
                authors.append(author_info)
        return authors
    
    def _determine_status(self, mal_status: str) -> str:
        """Map MAL status to unified status."""
        status_map = {
            "finished_airing": "completed",
            "currently_airing": "ongoing",
            "not_yet_aired": "upcoming",
            "finished": "completed",
            "currently_publishing": "ongoing",
            "not_yet_published": "upcoming"
        }
        return status_map.get(mal_status, mal_status)
    
    def _is_nsfw(self, node: Dict[str, Any]) -> bool:
        """Determine if content is NSFW based on MAL data."""
        nsfw_flag = node.get("nsfw", "white")
        return nsfw_flag in ["gray", "black"]
    
    def normalize_to_media_base(
        self, 
        raw_data: Dict[str, Any], 
        media_type: str = "anime"
    ) -> Dict[str, Any]:
        """Convert MAL response to MediaBase format.
        
        Args:
            raw_data: Raw MAL data (from node field)
            media_type: "anime" or "manga"
            
        Returns:
            MediaBase-compatible dictionary
        """
        node = raw_data.get("node", raw_data)
        
        mal_id = node.get("id")
        title = node.get("title")
        
        # Get main picture
        main_picture = None
        pictures = node.get("main_picture", {})
        if pictures:
            main_picture = pictures.get("large") or pictures.get("medium")
        
        # Get synopsis
        synopsis = node.get("synopsis")
        
        # Get score
        score = node.get("mean")
        
        # Get genres
        genres = self._extract_genres(node)
        
        # Get sub_type from media_type
        sub_type = node.get("media_type", "").upper() if media_type == "anime" else None
        
        # Map status
        mal_status = node.get("status", "")
        status = self._determine_status(mal_status)
        
        # Get release date
        release_date = node.get("start_date")
        
        # Build metadata
        metadata = {
            "source": "mal",
            "original_id": mal_id,
            "rank": node.get("rank"),
            "popularity": node.get("popularity"),
            "num_list_users": node.get("num_list_users"),
            "num_scoring_users": node.get("num_scoring_users"),
            "nsfw": self._is_nsfw(node),
            "sfw": not self._is_nsfw(node),
            "alternative_titles": node.get("alternative_titles", {}),
            "end_date": node.get("end_date"),
            "source_material": node.get("source"),
            "rating": node.get("rating"),
        }
        
        # Add media-specific fields
        if media_type == "anime":
            metadata["episodes"] = node.get("num_episodes")
            metadata["studios"] = self._extract_studios(node)
            metadata["broadcast"] = node.get("broadcast", {})
            metadata["season"] = node.get("start_season", {}).get("season")
            metadata["year"] = node.get("start_season", {}).get("year")
            metadata["average_episode_duration"] = node.get("average_episode_duration")
        else:  # manga
            metadata["chapters"] = node.get("num_chapters")
            metadata["volumes"] = node.get("num_volumes")
            metadata["authors"] = self._extract_authors(node)
        
        return {
            "media_id": f"{media_type}-{mal_id}",
            "title": title,
            "synopsis": synopsis,
            "main_picture": main_picture,
            "score": score,
            "genres": genres,
            "media_type": media_type,
            "sub_type": sub_type,
            "status": status,
            "release_date": release_date,
            "metadata": metadata,
            "source": "mal",
            "source_id": str(mal_id),
        }
