from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import List, Optional, Annotated
from app.models.media import RecommendationRequest, RecommendationResponse, MediaRecommendation
from app.services.recommender import RecommenderService
import logging

logger = logging.getLogger(__name__)

media_router = APIRouter()

# Dependency to get the initialized recommender service
async def get_recommender_service() -> RecommenderService:
    from main import recommender_service
    try:
        if not recommender_service.is_initialized:
            logger.info("Initializing recommender service on first request...")
            await recommender_service.initialize()
        return recommender_service
    except Exception as e:
        logger.error(f"Failed to initialize recommender service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender service is currently unavailable"
        )

# Type alias for the service dependency
RecommenderDep = Annotated[RecommenderService, Depends(get_recommender_service)]

@media_router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    service: RecommenderDep
):
    """
    Get personalized media recommendations using hybrid search (Content-based + Semantic).
    """
    try:
        return await service.get_recommendations(request)
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )

@media_router.get("/search", response_model=List[MediaRecommendation])
async def search_media(
    service: RecommenderDep,
    q: str = Query(..., min_length=1, description="Search query for media"),
    media_type: Optional[str] = Query(None, description="Filter by media type (anime, manga, etc.)"),
    limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    genres: Optional[List[str]] = Query(None, description="Filter by genres")
):
    """
    Search for media using semantic vector search.
    """
    try:
        return await service.search(q, limit=limit, offset=offset, genres=genres, media_type=media_type)
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed"
        )

@media_router.get("/genres", response_model=List[str])
async def get_genres(
    service: RecommenderDep,
    media_type: Optional[str] = Query(None, description="Filter genres by media type")
):
    """
    Get all available genres for the specified media type.
    """
    try:
        return service.get_all_genres(media_type=media_type)
    except Exception as e:
        logger.error(f"Error fetching genres: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch genres"
        )

@media_router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_data(
    service: RecommenderDep,
    batch_size: int = Query(100, ge=1, le=1000)
):
    """
    Trigger data ingestion into the vector database.
    This is an administrative operation.
    """
    try:
        # Run ingestion in the background
        import asyncio
        asyncio.create_task(service.ingest_data(batch_size=batch_size))
        return {"message": "Ingestion started in background"}
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start ingestion"
        )

@media_router.get("/health")
async def health_check(service: RecommenderDep):
    """
    Check the health of the service and its dependencies.
    """
    return await service.check_health()

@media_router.get("/trending", response_model=List[MediaRecommendation])
async def get_trending(
    service: RecommenderDep,
    limit: int = Query(12, ge=1, le=100),
    media_type: Optional[str] = Query(None)
):
    """
    Get trending media.
    """
    try:
        return await service.get_trending(limit=limit, media_type=media_type)
    except Exception as e:
        logger.error(f"Error fetching trending: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch trending media"
        )

@media_router.get("/manga/search", response_model=List[MediaRecommendation])
async def search_manga(
    service: RecommenderDep,
    q: str = Query(..., min_length=1, description="Search query for manga"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    genres: Optional[List[str]] = Query(None)
):
    """
    Search specifically for manga, manhwa, and manhua.
    """
    try:
        return await service.search(q, limit=limit, offset=offset, genres=genres, media_type="manga")
    except Exception as e:
        logger.error(f"Manga search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Manga search failed"
        )

@media_router.get("/movies/search", response_model=List[MediaRecommendation])
async def search_movies(
    service: RecommenderDep,
    q: str = Query(..., min_length=1, description="Search query for movies"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    genres: Optional[List[str]] = Query(None)
):
    """
    Search specifically for movies.
    """
    try:
        return await service.search(q, limit=limit, offset=offset, genres=genres, media_type="movie")
    except Exception as e:
        logger.error(f"Movie search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Movie search failed"
        )

@media_router.get("/tv/search", response_model=List[MediaRecommendation])
async def search_tv(
    service: RecommenderDep,
    q: str = Query(..., min_length=1, description="Search query for TV shows"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    genres: Optional[List[str]] = Query(None)
):
    """
    Search specifically for TV shows and dramas.
    """
    try:
        return await service.search(q, limit=limit, offset=offset, genres=genres, media_type="tv")
    except Exception as e:
        logger.error(f"TV search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TV search failed"
        )

@media_router.post("/cross-recommend", response_model=List[MediaRecommendation])
async def cross_recommend(
    service: RecommenderDep,
    media_id: str = Query(..., description="Source media ID (e.g., 'anime-123', 'manga-456')"),
    target_media_types: List[str] = Query(..., description="Target media types to recommend"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get cross-media recommendations (e.g., find movies similar to an anime).
    """
    try:
        return await service.get_cross_media_recommendations(
            media_id=media_id,
            target_media_types=target_media_types,
            top_n=limit
        )
    except Exception as e:
        logger.error(f"Cross-media recommendation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cross-media recommendation failed"
        )

@media_router.get("/{media_id}", response_model=MediaRecommendation)
async def get_media_details(
    media_id: int,
    service: RecommenderDep
):
    """
    Get detailed information about a specific media item by its ID.
    """
    try:
        item = await service.get_by_id(media_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Media with ID {media_id} not found"
            )
        return item
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching media details for {media_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch media details"
        )
