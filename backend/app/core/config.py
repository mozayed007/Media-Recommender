from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Any, Optional
import os

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Media Recommender"
    API_V1_STR: str = "/api/v1"
    
    # Data Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    DATA_PATH: str = "backend/data/raw/anime_Jan26.parquet"
    MULTI_MEDIA_DATA_PATH: str = "backend/data/processed/multi_media_v1.parquet"
    
    @property
    def absolute_data_path(self) -> str:
        if os.path.isabs(self.DATA_PATH):
            return self.DATA_PATH
        return os.path.join(self.BASE_DIR, self.DATA_PATH)
    
    @property
    def absolute_multi_media_path(self) -> str:
        if os.path.isabs(self.MULTI_MEDIA_DATA_PATH):
            return self.MULTI_MEDIA_DATA_PATH
        return os.path.join(self.BASE_DIR, self.MULTI_MEDIA_DATA_PATH)
    
    # Vector DB Settings
    VECTOR_DB_PROVIDER: str = "qdrant"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_PATH: Optional[str] = "backend/data/qdrant" # Default to local storage path
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "anime_embeddings_v2"

    @property
    def qdrant_storage_path(self) -> Optional[str]:
        if not self.QDRANT_PATH:
            return None
        if os.path.isabs(self.QDRANT_PATH):
            return self.QDRANT_PATH
        return os.path.join(self.BASE_DIR, self.QDRANT_PATH)
    
    # ML Settings
    EMBEDDING_MODEL: str = "google/embeddinggemma-300m"
    EMBEDDING_DIMENSION: int = 768
    HF_TOKEN: Optional[str] = None
    
    # MAL API
    MAL_CLIENT_ID: Optional[str] = None
    
    # Media API Keys
    TMDB_API_KEY: Optional[str] = None
    TMDB_READ_ACCESS_TOKEN: Optional[str] = None  # For TMDB v4 API
    MANGADEX_API_KEY: Optional[str] = None  # For authenticated requests
    
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
    
    # Data Quality Thresholds
    MIN_SYNOPSIS_LENGTH: int = 50
    MIN_SCORE_THRESHOLD: float = 0.0
    REQUIRE_GENRES: bool = False
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "openai:gpt-4o"
    
    # Frontend Configuration
    NEXT_PUBLIC_API_URL: str = "http://localhost:8000"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    # Feature Columns
    TEXT_FEATURES: List[str] = ["genres", "themes", "demographics", "studios"]
    NUMERIC_FEATURES: List[str] = ["score", "episodes"]
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore" # Allow extra fields in .env without crashing
    )

settings = Settings()
