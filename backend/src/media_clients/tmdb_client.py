from typing import List, Dict, Any, Optional
import aiohttp
import logging
from .base_client import MediaClient
from .utils import RateLimiter, with_retry, NotFoundError, APIError

logger = logging.getLogger(__name__)

# TMDB Genre ID to name mapping
TMDB_GENRE_MAP = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Sci-Fi",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
    # TV genres
    10759: "Action & Adventure",
    10762: "Kids",
    10763: "News",
    10764: "Reality",
    10765: "Sci-Fi & Fantasy",
    10766: "Soap",
    10767: "Talk",
    10768: "War & Politics",
}

class TMDBClient(MediaClient):
    """Client for The Movie Database (TMDB) API v3 and v4."""
    
    def __init__(self, api_key: str, read_access_token: Optional[str] = None, rate_limit: int = 40):
        """Initialize TMDB client.
        
        Args:
            api_key: TMDB API key (for v3 API)
            read_access_token: TMDB read access token (for v4 API)
            rate_limit: Requests per 10 seconds (default: 40)
        """
        super().__init__(api_key=api_key, base_url="https://api.themoviedb.org/3")
        self.rate_limiter = RateLimiter(rate=rate_limit, per=10.0)
        self.image_base_url = "https://image.tmdb.org/t/p/"
        self.read_access_token = read_access_token
        self.v4_base_url = "https://api.themoviedb.org/4"
        self.use_v4 = read_access_token is not None
    
    @with_retry(max_attempts=3)
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make rate-limited request to TMDB API.
        
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
        
        # Add API key to params
        if params is None:
            params = {}
        params["api_key"] = self.api_key
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 404:
                    raise NotFoundError(f"Resource not found: {url}")
                elif response.status == 429:
                    self.logger.warning("Rate limit exceeded, backing off...")
                    raise APIError("Rate limit exceeded")
                elif response.status == 401:
                    raise APIError("Invalid API key")
                elif response.status != 200:
                    raise APIError(f"API request failed with status {response.status}")
                
                return await response.json()
        except aiohttp.ClientError as e:
            self.logger.error(f"Request failed: {e}")
            raise APIError(f"Request failed: {e}")
    
    @with_retry(max_attempts=3)
    async def _make_v4_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make rate-limited request to TMDB v4 API using Bearer token.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response
        """
        if not self.read_access_token:
            raise APIError("v4 API requires read_access_token")
        
        await self.rate_limiter.acquire()
        
        if not self.session:
            await self.initialize()
        
        url = f"{self.v4_base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.read_access_token}"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 404:
                    raise NotFoundError(f"Resource not found: {url}")
                elif response.status == 429:
                    self.logger.warning("Rate limit exceeded, backing off...")
                    raise APIError("Rate limit exceeded")
                elif response.status == 401:
                    raise APIError("Invalid read access token")
                elif response.status != 200:
                    raise APIError(f"API request failed with status {response.status}")
                
                return await response.json()
        except aiohttp.ClientError as e:
            self.logger.error(f"Request failed: {e}")
            raise APIError(f"Request failed: {e}")
    
    def _map_genre_ids(self, genre_ids: List[int]) -> List[str]:
        """Map TMDB genre IDs to names.
        
        Args:
            genre_ids: List of genre IDs
            
        Returns:
            List of genre names
        """
        return [TMDB_GENRE_MAP.get(gid) for gid in genre_ids if TMDB_GENRE_MAP.get(gid)]
    
    async def get_by_filters(
        self,
        media_type: str = "movie",
        language: Optional[str] = None,
        region: Optional[str] = None,
        certification: Optional[str] = None,
        min_vote_count: int = 0,
        min_score: float = 0.0,
        max_score: float = 10.0,
        genres: Optional[List[int]] = None,
        year: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get movies or TV shows by advanced filters.
        
        Args:
            media_type: "movie" or "tv"
            language: Original language filter (e.g., "en", "ko", "ja")
            region: Region filter (e.g., "US", "KR")
            certification: Content rating filter (e.g., "PG-13", "R")
            min_vote_count: Minimum number of votes
            min_score: Minimum rating (0-10)
            max_score: Maximum rating (0-10)
            genres: List of genre IDs
            year: Release year
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of filtered media data
        """
        page = (offset // limit) + 1
        
        params = {
            "sort_by": "popularity.desc",
            "page": page,
            "language": "en-US",
        }
        
        # Apply filters
        if language:
            params["with_original_language"] = language
        
        if region:
            params["region"] = region
            params["watch_region"] = region
        
        if certification and media_type == "movie":
            params["certification_country"] = region or "US"
            params["certification.lte"] = certification
        
        if min_vote_count > 0:
            params["vote_count.gte"] = min_vote_count
        
        if min_score > 0:
            params["vote_average.gte"] = min_score
        
        if max_score < 10:
            params["vote_average.lte"] = max_score
        
        if genres:
            params["with_genres"] = ",".join(map(str, genres))
        
        if year:
            if media_type == "movie":
                params["primary_release_year"] = year
            else:
                params["first_air_date_year"] = year
        
        endpoint = f"/discover/{media_type}"
        response = await self._make_request(endpoint, params)
        
        results = response.get("results", [])
        start_idx = offset % limit
        return results[start_idx:start_idx + limit]
    
    async def get_detailed(self, media_id: str, media_type: str = "movie") -> Dict[str, Any]:
        """Get detailed movie or TV show information including credits and keywords.
        
        Args:
            media_id: TMDB ID
            media_type: "movie" or "tv"
            
        Returns:
            Detailed media data with credits, keywords, and watch providers
        """
        params = {
            "append_to_response": "credits,keywords,watch/providers,videos,images",
            "language": "en-US",
        }
        
        endpoint = f"/{media_type}/{media_id}"
        return await self._make_request(endpoint, params)
    
    async def get_season_details(
        self, 
        tv_id: str, 
        season_number: int
    ) -> Dict[str, Any]:
        """Get TV show season details.
        
        Args:
            tv_id: TMDB TV show ID
            season_number: Season number
            
        Returns:
            Season details with episodes
        """
        params = {
            "language": "en-US",
        }
        
        endpoint = f"/tv/{tv_id}/season/{season_number}"
        return await self._make_request(endpoint, params)
    
    async def search(self, query: str, limit: int = 20, offset: int = 0, media_type: str = "movie") -> List[Dict[str, Any]]:
        """Search for movies or TV shows.
        
        Args:
            query: Search query
            limit: Maximum results per page
            offset: Pagination offset (page number)
            media_type: "movie" or "tv"
            
        Returns:
            List of media data
        """
        page = (offset // limit) + 1
        
        params = {
            "query": query,
            "page": page,
            "language": "en-US",
        }
        
        endpoint = f"/search/{media_type}"
        response = await self._make_request(endpoint, params)
        
        results = response.get("results", [])
        # Return only the requested slice
        start_idx = offset % limit
        return results[start_idx:start_idx + limit]
    
    async def get_by_id(self, media_id: str, media_type: str = "movie") -> Dict[str, Any]:
        """Get movie or TV show details by ID.
        
        Args:
            media_id: TMDB ID
            media_type: "movie" or "tv"
            
        Returns:
            Media data
        """
        params = {
            "append_to_response": "credits",
            "language": "en-US",
        }
        
        endpoint = f"/{media_type}/{media_id}"
        return await self._make_request(endpoint, params)
    
    async def get_popular(self, limit: int = 100, offset: int = 0, media_type: str = "movie") -> List[Dict[str, Any]]:
        """Get popular movies or TV shows.
        
        Args:
            limit: Maximum results per page
            offset: Pagination offset (page number)
            media_type: "movie" or "tv"
            
        Returns:
            List of media data
        """
        page = (offset // limit) + 1
        
        params = {
            "page": page,
            "language": "en-US",
        }
        
        endpoint = f"/{media_type}/popular"
        response = await self._make_request(endpoint, params)
        
        results = response.get("results", [])
        start_idx = offset % limit
        return results[start_idx:start_idx + limit]
    
    async def get_asian_dramas(self, country_code: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get TV shows from specific Asian countries.
        
        Args:
            country_code: Country code (KR, JP, CN)
            limit: Maximum results per page
            offset: Pagination offset
            
        Returns:
            List of TV show data
        """
        page = (offset // limit) + 1
        
        params = {
            "with_origin_country": country_code,
            "sort_by": "popularity.desc",
            "page": page,
            "language": "en-US",
            "vote_count.gte": 10,  # Filter out obscure content
        }
        
        endpoint = "/discover/tv"
        response = await self._make_request(endpoint, params)
        
        results = response.get("results", [])
        start_idx = offset % limit
        return results[start_idx:start_idx + limit]
    
    def _get_poster_url(self, poster_path: Optional[str], size: str = "w500") -> Optional[str]:
        """Get full poster URL.
        
        Args:
            poster_path: Poster path from TMDB
            size: Image size (w500, original, etc.)
            
        Returns:
            Full poster URL or None
        """
        if poster_path:
            return f"{self.image_base_url}{size}{poster_path}"
        return None
    
    def _extract_genres(self, genres: List[Dict[str, Any]]) -> List[str]:
        """Extract genre names from TMDB genre objects."""
        return [genre.get("name") for genre in genres if genre.get("name")]
    
    def _determine_sub_type(self, data: Dict[str, Any], media_type: str) -> str:
        """Determine sub_type based on origin country and media type."""
        if media_type == "movie":
            return "Movie"
        
        # For TV shows, check origin country
        origin_countries = data.get("origin_country", [])
        if "KR" in origin_countries:
            return "K-Drama"
        elif "JP" in origin_countries:
            return "J-Drama"
        elif "CN" in origin_countries:
            return "C-Drama"
        else:
            return "TV Series"
    
    def _extract_cast(self, credits: Dict[str, Any], limit: int = 5) -> List[str]:
        """Extract cast names from credits."""
        cast = credits.get("cast", [])
        return [person.get("name") for person in cast[:limit] if person.get("name")]
    
    def normalize_to_media_base(self, raw_data: Dict[str, Any], media_type: str = "movie") -> Dict[str, Any]:
        """Convert TMDB response to MediaBase format.
        
        Args:
            raw_data: Raw TMDB data
            media_type: "movie" or "tv"
            
        Returns:
            MediaBase-compatible dictionary
        """
        tmdb_id = raw_data.get("id")
        
        # Title differs between movies and TV
        title = raw_data.get("title") if media_type == "movie" else raw_data.get("name")
        synopsis = raw_data.get("overview")
        poster_url = self._get_poster_url(raw_data.get("poster_path"))
        score = raw_data.get("vote_average")
        genres = self._extract_genres(raw_data.get("genres", []))
        
        # If genres not in detailed response, try genre_ids
        if not genres and "genre_ids" in raw_data:
            genres = [str(gid) for gid in raw_data.get("genre_ids", [])]
        
        sub_type = self._determine_sub_type(raw_data, media_type)
        
        # Status
        status = raw_data.get("status", "unknown")
        
        # Release date
        release_date = raw_data.get("release_date") if media_type == "movie" else raw_data.get("first_air_date")
        
        # Extract cast if available
        cast = []
        if "credits" in raw_data:
            cast = self._extract_cast(raw_data["credits"])
        
        # Additional metadata
        metadata = {
            "source": "tmdb",
            "original_id": tmdb_id,
            "original_title": raw_data.get("original_title") or raw_data.get("original_name"),
            "original_language": raw_data.get("original_language"),
            "origin_country": raw_data.get("origin_country", []),
            "popularity": raw_data.get("popularity"),
            "vote_count": raw_data.get("vote_count"),
            "adult": raw_data.get("adult", False),
        }
        
        if cast:
            metadata["cast"] = cast
        
        if media_type == "tv":
            metadata["number_of_seasons"] = raw_data.get("number_of_seasons")
            metadata["number_of_episodes"] = raw_data.get("number_of_episodes")
        else:
            metadata["runtime"] = raw_data.get("runtime")
        
        return {
            "media_id": f"{media_type}-{tmdb_id}",
            "title": title,
            "synopsis": synopsis,
            "main_picture": poster_url,
            "score": score,
            "genres": genres,
            "media_type": media_type,
            "sub_type": sub_type,
            "status": status,
            "release_date": release_date,
            "metadata": metadata,
        }
