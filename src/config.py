"""
Configuration module for the Media Recommender system.
Centralizes all configuration to ensure consistency across the application.
"""
import os
import platform

# Base data paths (platform-independent)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Files
ANIME_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "anime_mal_Aug24.parquet")
PROCESSED_PICKLE_FILE = os.path.join(PROCESSED_DATA_DIR, "milvus_anime_Aug24.pkl")

# Column names standardization
COLUMN_NAMES = {
    "id": "anime_id",
    "title": "title",
    "description": "synopsis",
    "genres": "genres",
    "themes": "themes",
    "studios": "studios",
    "demographics": "demographics",
    "score": "score",
    "episodes": "episodes",
    "media_type": "media_type",
    "rating": "rating"
}

# Content feature columns
CONTENT_FEATURES = {
    "text": ["genres", "themes", "demographics", "studios"],
    "numeric": ["score", "episodes"]
}

# Vector database configuration
VECTOR_DB = {
    "type": "milvus",
    "host": "localhost",
    "port": "19530",
    "uri": "http://localhost:19530",
    "collection_name": "anime_embeddings_prod",
    "dimension": 384
}

# Embedding model configuration
EMBEDDING_MODEL = {
    "name": "all-MiniLM-L6-v2",
    "dimension": 384
}

# System settings
BATCH_SIZE = 64
CACHE_SIZE = 1024

def get_db_uri():
    """Get the database URI with the correct format based on host and port."""
    return f"http://{VECTOR_DB['host']}:{VECTOR_DB['port']}"
