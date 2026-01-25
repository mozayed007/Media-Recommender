from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AnimeBase(BaseModel):
    anime_id: int
    title: str
    synopsis: Optional[str] = None
    main_picture: Optional[str] = None
    score: Optional[float] = None
    genres: List[str] = []
    studios: List[str] = []
    type: Optional[str] = None
    episodes: Optional[int] = None
    status: Optional[str] = None

class AnimeRecommendation(AnimeBase):
    similarity_score: float

class RecommendationRequest(BaseModel):
    query: Optional[str] = None
    anime_id: Optional[int] = None
    top_n: int = 10
    use_rag: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)

class RecommendationResponse(BaseModel):
    recommendations: List[AnimeRecommendation]
    metadata: Dict[str, Any] = Field(default_factory=dict)
