import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from app.api.endpoints import media_router
from app.services.recommender import RecommenderService
from app.core.config import settings
from app.core.logging_config import setup_logging

# Setup logging
logger = setup_logging("api")

# Global state for service
recommender_service = RecommenderService(provider=settings.VECTOR_DB_PROVIDER)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the Recommender Service
    logger.info(f"Starting up: Initializing Recommender Service with {settings.VECTOR_DB_PROVIDER}...")
    try:
        await recommender_service.initialize()
        logger.info("Startup complete: Recommender Service is ready.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # In production, you might want to prevent startup if core service fails
    
    yield
    
    # Shutdown: Cleanup resources if needed
    logger.info("Shutting down: Recommender Service.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Inclusive Semantic Media Recommender System (Anime, Manga, Dramas, Novels, TV, Movies, Games, Music)",
    version="1.1.0",
    lifespan=lifespan
)

# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Invalid request parameters"},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "message": str(exc)},
    )

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(media_router, prefix=f"{settings.API_V1_STR}/media", tags=["media"])
# Keep anime alias for now to avoid breaking existing frontend during migration
app.include_router(media_router, prefix=f"{settings.API_V1_STR}/anime", tags=["anime"])

@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """
    Comprehensive health check including vector DB connectivity.
    """
    health_data = await recommender_service.check_health()
    status_code = status.HTTP_200_OK if health_data["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content=health_data
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
