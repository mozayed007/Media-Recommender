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
    
    @property
    def absolute_data_path(self) -> str:
        if os.path.isabs(self.DATA_PATH):
            return self.DATA_PATH
        return os.path.join(self.BASE_DIR, self.DATA_PATH)
    
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
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "openai:gpt-4o"
    
    # Frontend Configuration
    NEXT_PUBLIC_API_URL: str = "http://localhost:8000"
    
    # Feature Columns
    TEXT_FEATURES: List[str] = ["genres", "themes", "demographics", "studios"]
    NUMERIC_FEATURES: List[str] = ["score", "episodes"]
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore" # Allow extra fields in .env without crashing
    )

settings = Settings()
