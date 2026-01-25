from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class MediaBase(BaseModel):
    media_id: int
    title: str
    synopsis: Optional[str] = None
    main_picture: Optional[str] = None
    score: Optional[float] = None
    genres: List[str] = []
    media_type: str = "unknown" # e.g., anime, manga, drama, movie, game, music
    sub_type: Optional[str] = None # e.g., TV, Movie, OVA (for anime), Novel, Manhwa (for manga)
    status: Optional[str] = None
    release_date: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict) # Catch-all for extra fields like studios, authors, etc.

class MediaRecommendation(MediaBase):
    similarity_score: float

class RecommendationRequest(BaseModel):
    query: Optional[str] = None
    media_id: Optional[int] = None
    media_type: Optional[str] = None # Filter by media type if needed
    top_n: int = 10
    use_rag: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)

class RecommendationResponse(BaseModel):
    recommendations: List[MediaRecommendation]
    metadata: Dict[str, Any] = Field(default_factory=dict)
