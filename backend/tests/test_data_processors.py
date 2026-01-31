import pytest
import pandas as pd
from src.data_processing.manga_processor import MangaProcessor
from src.data_processing.tmdb_processor import TMDBProcessor

def test_manga_processor_clean_synopsis():
    """Test manga synopsis cleaning."""
    processor = MangaProcessor()
    
    # Test HTML removal
    dirty = "<p>This is a <b>test</b> synopsis.</p>"
    clean = processor._clean_synopsis(dirty)
    assert clean == "This is a test synopsis."
    
    # Test whitespace normalization
    dirty = "This   has    extra    spaces"
    clean = processor._clean_synopsis(dirty)
    assert clean == "This has extra spaces"
    
    # Test prefix removal
    dirty = "Synopsis: This is the actual synopsis"
    clean = processor._clean_synopsis(dirty)
    assert clean == "This is the actual synopsis"

def test_manga_processor_process_batch():
    """Test manga batch processing."""
    processor = MangaProcessor()
    
    raw_data = [
        {
            "media_id": "manga-1",
            "title": "Test Manga 1",
            "synopsis": "<p>Test synopsis</p>",
            "genres": ["Action", "Adventure"],
            "media_type": "manga",
            "sub_type": "Shounen",
            "main_picture": "http://example.com/cover.jpg",
            "score": None,
            "status": "ongoing",
            "release_date": "2020",
            "metadata": {}
        },
        {
            "media_id": "manga-2",
            "title": "Test Manga 2",
            "synopsis": None,  # Missing synopsis
            "genres": ["Romance"],
            "media_type": "manga",
            "sub_type": "Shoujo",
            "main_picture": None,
            "score": None,
            "status": "completed",
            "release_date": "2019",
            "metadata": {}
        }
    ]
    
    df = processor.process_batch(raw_data)
    
    assert len(df) == 2
    assert df.iloc[0]["synopsis"] == "Test synopsis"
    # Second entry should have fallback synopsis
    assert "Test Manga 2" in df.iloc[1]["synopsis"]

def test_manga_processor_validate():
    """Test manga data validation."""
    processor = MangaProcessor()
    
    data = [
        {
            "media_id": "manga-1",
            "title": "Valid Manga",
            "synopsis": "Has synopsis",
            "genres": ["Action"],
            "media_type": "manga",
            "sub_type": "Shounen",
        },
        {
            "media_id": "manga-2",
            "title": "Invalid Manga",
            "synopsis": None,  # No synopsis
            "genres": [],  # No genres
            "media_type": "manga",
            "sub_type": "Shounen",
        },
        {
            "media_id": "manga-3",
            "title": "Another Valid",
            "synopsis": None,
            "genres": ["Romance"],  # Has genres
            "media_type": "manga",
            "sub_type": "Shoujo",
        }
    ]
    
    df = pd.DataFrame(data)
    validated = processor.validate_data(df)
    
    # Should keep entries with synopsis OR genres
    assert len(validated) == 2
    assert "manga-1" in validated["media_id"].values
    assert "manga-3" in validated["media_id"].values

def test_tmdb_processor_clean_synopsis():
    """Test TMDB synopsis cleaning."""
    processor = TMDBProcessor()
    
    # Test whitespace normalization
    dirty = "This   has    extra    spaces"
    clean = processor._clean_synopsis(dirty)
    assert clean == "This has extra spaces"
    
    # Test prefix removal
    dirty = "Overview: This is the actual overview"
    clean = processor._clean_synopsis(dirty)
    assert clean == "This is the actual overview"

def test_tmdb_processor_process_batch():
    """Test TMDB batch processing."""
    processor = TMDBProcessor(min_vote_count=50, min_score=5.0)
    
    raw_data = [
        {
            "media_id": "movie-1",
            "title": "Test Movie",
            "synopsis": "Test synopsis",
            "genres": ["Action", "Thriller"],
            "media_type": "movie",
            "sub_type": "Movie",
            "main_picture": "http://example.com/poster.jpg",
            "score": 7.5,
            "status": "Released",
            "release_date": "2020-01-01",
            "metadata": {"vote_count": 1000}
        }
    ]
    
    df = processor.process_batch(raw_data)
    
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Test Movie"
    assert df.iloc[0]["vote_count"] == 1000

def test_tmdb_processor_validate():
    """Test TMDB data validation."""
    processor = TMDBProcessor(min_vote_count=50, min_score=5.0)
    
    data = [
        {
            "media_id": "movie-1",
            "title": "Good Movie",
            "synopsis": "Has synopsis",
            "genres": ["Action"],
            "media_type": "movie",
            "score": 7.5,
            "vote_count": 1000,
        },
        {
            "media_id": "movie-2",
            "title": "Low Votes",
            "synopsis": "Has synopsis",
            "genres": ["Drama"],
            "media_type": "movie",
            "score": 8.0,
            "vote_count": 10,  # Too few votes
        },
        {
            "media_id": "movie-3",
            "title": "Low Score",
            "synopsis": "Has synopsis",
            "genres": ["Comedy"],
            "media_type": "movie",
            "score": 3.0,  # Too low score
            "vote_count": 1000,
        },
        {
            "media_id": "movie-4",
            "title": "No Synopsis",
            "synopsis": None,  # Missing synopsis
            "genres": ["Horror"],
            "media_type": "movie",
            "score": 7.0,
            "vote_count": 500,
        }
    ]
    
    df = pd.DataFrame(data)
    validated = processor.validate_data(df)
    
    # Should only keep movie-1
    assert len(validated) == 1
    assert validated.iloc[0]["media_id"] == "movie-1"

def test_tmdb_processor_separate_asian_dramas():
    """Test separation of Asian dramas from Western content."""
    processor = TMDBProcessor()
    
    data = [
        {
            "media_id": "tv-1",
            "title": "K-Drama",
            "media_type": "tv",
            "metadata": {"origin_country": ["KR"]}
        },
        {
            "media_id": "tv-2",
            "title": "J-Drama",
            "media_type": "tv",
            "metadata": {"origin_country": ["JP"]}
        },
        {
            "media_id": "tv-3",
            "title": "US Show",
            "media_type": "tv",
            "metadata": {"origin_country": ["US"]}
        },
        {
            "media_id": "movie-1",
            "title": "Korean Movie",
            "media_type": "movie",
            "metadata": {"origin_country": ["KR"]}
        }
    ]
    
    df = pd.DataFrame(data)
    asian_dramas, western = processor.separate_asian_dramas(df)
    
    # Should have 2 Asian dramas (KR and JP TV shows)
    assert len(asian_dramas) == 2
    # Should have 2 Western/other (US show and KR movie)
    assert len(western) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
