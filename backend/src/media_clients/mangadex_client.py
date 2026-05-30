from typing import List, Dict, Any, Optional
import aiohttp
import logging
from .base_client import MediaClient
from .utils import RateLimiter, with_retry, NotFoundError, APIError

logger = logging.getLogger(__name__)

# MangaDex tag IDs to unified genre mapping
MANGADEX_GENRE_MAP = {
    "391b0423-d847-456f-aff0-8b0cfc03066b": "Action",
    "87cc87cd-a395-47af-b27a-93258283bbc6": "Adventure",
    "5920b825-4181-4a17-beeb-9918b0ff7a30": "Boys Love",
    "4d32cc48-9f00-4cca-9b5a-a839f0764984": "Comedy",
    "b9af3a63-f058-46de-a9a0-e0c13906197a": "Drama",
    "cdc58593-87dd-415e-bbc0-2ec27bf404cc": "Fantasy",
    "a3c67850-4684-404e-9b7f-c69850ee5da6": "Girls Love",
    "3e2b8dae-350e-4ab8-a8ce-016e844b9f0d": "Historical",
    "cdad7e68-1419-41dd-bdce-27753074a640": "Horror",
    "ace04993-3797-43da-b767-b0a25f89f4b7": "Isekai",
    "acc803a4-c95a-4c22-86fc-eb6b582d82a2": "Martial Arts",
    "50880a9d-5440-4732-9afb-8f457127e836": "Mecha",
    "ee968100-4191-4968-93d3-f82d72be7e46": "Medical",
    "489dd859-9b61-4c37-af75-5b18e88daafc": "Music",
    "92d6d951-ca5e-429c-ac78-451071cbf064": "Mystery",
    "0234a31e-a729-4e28-9d6a-3f87c4966b9e": "Psychological",
    "423e2eae-a7a2-4a8b-ac03-a8351462d71d": "Romance",
    "256c8bd9-4904-4360-bf4f-508a76d67183": "Sci-Fi",
    "e5301a23-ebd9-49dd-a0cb-2add944c7fe9": "Slice of Life",
    "69964a64-2f90-4d33-beeb-f3ed2875eb4c": "Sports",
    "eabc5b4c-6aff-42f3-b657-3e90cbd00b75": "Supernatural",
    "f8f62932-27da-4fe4-8ee1-6779a8c5edba": "Thriller",
}

# MangaDex demographics
DEMOGRAPHICS = {
    "shounen": "Shounen",
    "shoujo": "Shoujo",
    "seinen": "Seinen",
    "josei": "Josei",
}

class MangaDexClient(MediaClient):
    """Client for MangaDex API with authentication support."""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 5):
        """Initialize MangaDex client.
        
        Args:
            api_key: API key for authenticated requests (increases rate limit to 10/s)
            rate_limit: Requests per second (5 unauthenticated, 10 authenticated)
        """
        super().__init__(base_url="https://api.mangadex.org")
        self.api_key = api_key
        # Authenticated users get higher rate limits
        effective_rate_limit = 10 if api_key else rate_limit
        self.rate_limiter = RateLimiter(rate=effective_rate_limit, per=1.0)
        self.session_token: Optional[str] = None
    
    @with_retry(max_attempts=3)
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make rate-limited request to MangaDex API.
        
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
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with self.session.get(url, params=params, headers=headers if headers else None) as response:
                if response.status == 404:
                    raise NotFoundError(f"Resource not found: {url}")
                elif response.status == 429:
                    self.logger.warning("Rate limit exceeded, backing off...")
                    raise APIError("Rate limit exceeded")
                elif response.status != 200:
                    raise APIError(f"API request failed with status {response.status}")
                
                return await response.json()
        except aiohttp.ClientError as e:
            self.logger.error(f"Request failed: {e}")
            raise APIError(f"Request failed: {e}")
    
    async def search(self, query: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Search for manga by title.
        
        Args:
            query: Search query
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of manga data
        """
        params = {
            "title": query,
            "limit": limit,
            "offset": offset,
            "includes[]": ["cover_art", "author", "artist"],
            "contentRating[]": ["safe", "suggestive"],
            "order[relevance]": "desc",
        }
        
        response = await self._make_request("/manga", params)
        return response.get("data", [])
    
    async def get_by_id(self, media_id: str) -> Dict[str, Any]:
        """Get manga details by ID.
        
        Args:
            media_id: MangaDex manga ID
            
        Returns:
            Manga data
        """
        params = {
            "includes[]": ["cover_art", "author", "artist"],
        }
        
        response = await self._make_request(f"/manga/{media_id}", params)
        return response.get("data", {})
    
    async def get_popular(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get popular manga sorted by follows.
        
        Args:
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of manga data
        """
        params = {
            "limit": limit,
            "offset": offset,
            "includes[]": ["cover_art", "author", "artist"],
            "contentRating[]": ["safe", "suggestive"],
            "order[followedCount]": "desc",
            "hasAvailableChapters": "true",
        }
        
        response = await self._make_request("/manga", params)
        return response.get("data", [])
    
    async def get_statistics(self, manga_id: str) -> Dict[str, Any]:
        """Get manga statistics including follows, ratings, and chapter info.
        
        Args:
            manga_id: MangaDex manga ID
            
        Returns:
            Statistics data including follows, rating, and chapter info
        """
        response = await self._make_request(f"/statistics/manga/{manga_id}")
        stats = response.get("statistics", {}).get(manga_id, {})
        
        return {
            "follows": stats.get("follows", 0),
            "rating_average": stats.get("rating", {}).get("average", 0),
            "rating_bayesian": stats.get("rating", {}).get("bayesian", 0),
            "total_ratings": stats.get("rating", {}).get("count", 0) if stats.get("rating") else 0,
            "comments": stats.get("comments", 0),
        }
    
    async def get_by_filters(
        self,
        limit: int = 100,
        offset: int = 0,
        publication_demographic: Optional[List[str]] = None,
        original_language: Optional[List[str]] = None,
        content_rating: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
        has_available_chapters: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get manga with advanced filtering.
        
        Args:
            limit: Maximum results
            offset: Pagination offset
            publication_demographic: Filter by demographic (shounen, shoujo, seinen, josei)
            original_language: Filter by original language (ja, ko, zh for manga/manhwa/manhua)
            content_rating: Filter by content rating (safe, suggestive, erotica, pornographic)
            status: Filter by status (ongoing, completed, hiatus, cancelled)
            has_available_chapters: Only include manga with available chapters
            
        Returns:
            List of filtered manga data
        """
        params = {
            "limit": limit,
            "offset": offset,
            "includes[]": ["cover_art", "author", "artist"],
            "order[followedCount]": "desc",
        }
        
        if publication_demographic:
            params["publicationDemographic[]"] = publication_demographic
        
        if original_language:
            params["originalLanguage[]"] = original_language
        
        if content_rating:
            params["contentRating[]"] = content_rating
        else:
            # Default to safe content
            params["contentRating[]"] = ["safe", "suggestive"]
        
        if status:
            params["status[]"] = status
        
        if has_available_chapters:
            params["hasAvailableChapters"] = "true"
        
        response = await self._make_request("/manga", params)
        return response.get("data", [])
    
    async def get_author_details(self, author_id: str) -> Dict[str, Any]:
        """Get detailed author information including other works.
        
        Args:
            author_id: MangaDex author/artist ID
            
        Returns:
            Author details with biography and social links
        """
        response = await self._make_request(f"/author/{author_id}")
        data = response.get("data", {})
        attributes = data.get("attributes", {})
        
        return {
            "id": data.get("id"),
            "name": attributes.get("name"),
            "biography": attributes.get("biography", {}),
            "twitter": attributes.get("twitter"),
            "pixiv": attributes.get("pixiv"),
            "website": attributes.get("website"),
        }
    
    def _extract_title(self, attributes: Dict[str, Any]) -> str:
        """Extract best available title."""
        title_data = attributes.get("title", {})
        
        # Priority: English > Romanized > Japanese > First available
        if "en" in title_data:
            return title_data["en"]
        elif "ja-ro" in title_data:
            return title_data["ja-ro"]
        elif "ja" in title_data:
            return title_data["ja"]
        else:
            # Return first available title
            return next(iter(title_data.values()), "Unknown Title")
    
    def _extract_synopsis(self, attributes: Dict[str, Any]) -> Optional[str]:
        """Extract best available synopsis."""
        description_data = attributes.get("description", {})
        
        # Priority: English > First available
        if "en" in description_data:
            return description_data["en"]
        else:
            return next(iter(description_data.values()), None)
    
    def _extract_cover_url(self, relationships: List[Dict[str, Any]], manga_id: str) -> Optional[str]:
        """Extract cover art URL from relationships."""
        for rel in relationships:
            if rel.get("type") == "cover_art":
                filename = rel.get("attributes", {}).get("fileName")
                if filename:
                    return f"https://uploads.mangadex.org/covers/{manga_id}/{filename}.512.jpg"
        return None
    
    def _map_genres(self, tags: List[Dict[str, Any]]) -> List[str]:
        """Map MangaDex tags to unified genres."""
        genres = []
        for tag in tags:
            tag_id = tag.get("id")
            if tag_id in MANGADEX_GENRE_MAP:
                genre = MANGADEX_GENRE_MAP[tag_id]
                if genre not in genres:
                    genres.append(genre)
        return genres
    
    def _extract_authors(self, relationships: List[Dict[str, Any]]) -> List[str]:
        """Extract author names from relationships."""
        authors = []
        for rel in relationships:
            if rel.get("type") in ["author", "artist"]:
                name = rel.get("attributes", {}).get("name")
                if name and name not in authors:
                    authors.append(name)
        return authors
    
    def normalize_to_media_base(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MangaDex response to MediaBase format.
        
        Args:
            raw_data: Raw MangaDex manga data
            
        Returns:
            MediaBase-compatible dictionary
        """
        manga_id = raw_data.get("id")
        attributes = raw_data.get("attributes", {})
        relationships = raw_data.get("relationships", [])
        
        title = self._extract_title(attributes)
        synopsis = self._extract_synopsis(attributes)
        cover_url = self._extract_cover_url(relationships, manga_id)
        genres = self._map_genres(attributes.get("tags", []))
        authors = self._extract_authors(relationships)
        
        # Determine sub_type from publication demographic
        demographic = attributes.get("publicationDemographic")
        sub_type = DEMOGRAPHICS.get(demographic, "Manga")
        
        # Determine if content is SFW
        content_rating = attributes.get("contentRating", "safe")
        sfw = content_rating in ["safe", "suggestive"]
        
        # Get status
        status = attributes.get("status", "unknown")
        
        # Get year from first available date
        year = attributes.get("year")
        
        return {
            "media_id": f"manga-{manga_id}",
            "title": title,
            "synopsis": synopsis,
            "main_picture": cover_url,
            "score": None,  # MangaDex doesn't provide ratings
            "genres": genres,
            "media_type": "manga",
            "sub_type": sub_type,
            "status": status,
            "release_date": str(year) if year else None,
            "metadata": {
                "source": "mangadex",
                "original_id": manga_id,
                "authors": authors,
                "content_rating": content_rating,
                "sfw": sfw,
                "demographic": demographic,
                "original_language": attributes.get("originalLanguage"),
            }
        }
