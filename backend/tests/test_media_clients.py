import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.media_clients.mangadex_client import MangaDexClient
from src.media_clients.tmdb_client import TMDBClient
from src.media_clients.utils import RateLimiter

@pytest.mark.asyncio
async def test_rate_limiter():
    """Test rate limiter functionality."""
    limiter = RateLimiter(rate=2, per=1.0)
    
    import time
    start = time.time()
    
    # Should allow 2 requests immediately
    await limiter.acquire()
    await limiter.acquire()
    
    # Third request should be delayed
    await limiter.acquire()
    
    elapsed = time.time() - start
    assert elapsed >= 0.5, "Rate limiter should delay third request"

@pytest.mark.asyncio
async def test_mangadex_client_initialization():
    """Test MangaDex client initialization."""
    client = MangaDexClient(rate_limit=5)
    
    assert client.base_url == "https://api.mangadex.org"
    assert client.rate_limiter is not None
    
    await client.initialize()
    assert client.session is not None
    
    await client.close()
    assert client.session is None

@pytest.mark.asyncio
async def test_mangadex_normalize():
    """Test MangaDex data normalization."""
    client = MangaDexClient()
    
    raw_data = {
        "id": "test-manga-id",
        "attributes": {
            "title": {"en": "Test Manga"},
            "description": {"en": "Test description"},
            "tags": [],
            "publicationDemographic": "shounen",
            "status": "ongoing",
            "year": 2020,
            "contentRating": "safe",
        },
        "relationships": []
    }
    
    normalized = client.normalize_to_media_base(raw_data)
    
    assert normalized["media_id"] == "manga-test-manga-id"
    assert normalized["title"] == "Test Manga"
    assert normalized["synopsis"] == "Test description"
    assert normalized["media_type"] == "manga"
    assert normalized["sub_type"] == "Shounen"
    assert normalized["metadata"]["sfw"] == True

@pytest.mark.asyncio
async def test_tmdb_client_initialization():
    """Test TMDB client initialization."""
    client = TMDBClient(api_key="test_key", rate_limit=40)
    
    assert client.base_url == "https://api.themoviedb.org/3"
    assert client.api_key == "test_key"
    assert client.rate_limiter is not None
    
    await client.initialize()
    assert client.session is not None
    
    await client.close()

@pytest.mark.asyncio
async def test_tmdb_normalize_movie():
    """Test TMDB movie data normalization."""
    client = TMDBClient(api_key="test_key")
    
    raw_data = {
        "id": 550,
        "title": "Fight Club",
        "overview": "A ticking-time-bomb insomniac...",
        "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        "vote_average": 8.4,
        "genres": [{"id": 18, "name": "Drama"}],
        "release_date": "1999-10-15",
        "status": "Released",
        "original_language": "en",
        "origin_country": ["US"],
        "popularity": 61.416,
        "vote_count": 26280,
        "adult": False,
    }
    
    normalized = client.normalize_to_media_base(raw_data, media_type="movie")
    
    assert normalized["media_id"] == "movie-550"
    assert normalized["title"] == "Fight Club"
    assert normalized["media_type"] == "movie"
    assert normalized["sub_type"] == "Movie"
    assert normalized["score"] == 8.4
    assert "Drama" in normalized["genres"]

@pytest.mark.asyncio
async def test_tmdb_normalize_kdrama():
    """Test TMDB K-Drama data normalization."""
    client = TMDBClient(api_key="test_key")
    
    raw_data = {
        "id": 12345,
        "name": "Test K-Drama",
        "overview": "A Korean drama...",
        "poster_path": "/test.jpg",
        "vote_average": 8.5,
        "genres": [{"id": 18, "name": "Drama"}],
        "first_air_date": "2020-01-01",
        "status": "Ended",
        "original_language": "ko",
        "origin_country": ["KR"],
        "popularity": 50.0,
        "vote_count": 1000,
        "adult": False,
        "number_of_seasons": 1,
        "number_of_episodes": 16,
    }
    
    normalized = client.normalize_to_media_base(raw_data, media_type="tv")
    
    assert normalized["media_id"] == "tv-12345"
    assert normalized["title"] == "Test K-Drama"
    assert normalized["media_type"] == "tv"
    assert normalized["sub_type"] == "K-Drama"
    assert normalized["metadata"]["number_of_seasons"] == 1
    assert normalized["metadata"]["number_of_episodes"] == 16

@pytest.mark.asyncio
async def test_mangadex_search_mock():
    """Test MangaDex search with mocked response."""
    client = MangaDexClient()
    
    mock_response = {
        "data": [
            {
                "id": "manga1",
                "attributes": {
                    "title": {"en": "Test Manga 1"},
                    "description": {"en": "Description 1"},
                    "tags": [],
                    "publicationDemographic": "shounen",
                    "status": "ongoing",
                    "contentRating": "safe",
                },
                "relationships": []
            }
        ]
    }
    
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        
        results = await client.search("test", limit=10)
        
        assert len(results) == 1
        assert results[0]["id"] == "manga1"
        mock_request.assert_called_once()

@pytest.mark.asyncio
async def test_tmdb_search_mock():
    """Test TMDB search with mocked response."""
    client = TMDBClient(api_key="test_key")
    
    mock_response = {
        "results": [
            {
                "id": 550,
                "title": "Fight Club",
                "overview": "Test overview",
                "poster_path": "/test.jpg",
                "vote_average": 8.4,
                "genre_ids": [18],
            }
        ]
    }
    
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        
        results = await client.search("Fight Club", media_type="movie")
        
        assert len(results) == 1
        assert results[0]["id"] == 550
        mock_request.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
