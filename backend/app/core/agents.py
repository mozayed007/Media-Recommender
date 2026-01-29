import logging
import json
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from app.core.config import settings

logger = logging.getLogger(__name__)

class QueryIntent(BaseModel):
    """Structured interpretation of a user's recommendation query."""
    search_terms: str = Field(description="The primary search terms for vector search")
    genres: List[str] = Field(default_factory=list, description="Specific genres mentioned")
    min_score: Optional[float] = Field(None, description="Minimum score if mentioned")
    is_recommendation: bool = Field(True, description="Whether the user is asking for recommendations or just searching")

async def interpret_query(query: str) -> QueryIntent:
    """
    Interpret user query using the latest Google GenAI SDK (google-genai).
    """
    if not settings.GEMINI_API_KEY:
        logger.warning("No Gemini API key found. Using rule-based fallback.")
        return _fallback_intent(query)

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        # Use gemini-1.5-flash which is a robust stable model
        model_id = 'gemini-3-flash-preview' 
        
        prompt = (
            "You are an expert anime recommendation assistant. "
            "Your task is to parse user queries into structured JSON for a recommendation system. "
            "Extract search terms, genres, and any specific constraints like minimum scores. "
            f"User Query: {query}"
        )

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QueryIntent
                )
            )

            # The SDK can parse the response directly into the Pydantic model
            if hasattr(response, 'parsed') and response.parsed:
                return response.parsed
            
            # Fallback to manual parsing if .parsed is not available
            data = json.loads(response.text)
            return QueryIntent(**data)
        except Exception as api_err:
            if "429" in str(api_err) or "RESOURCE_EXHAUSTED" in str(api_err):
                logger.warning(f"Gemini API quota exceeded (429). Falling back to rule-based: {api_err}")
            else:
                logger.error(f"Gemini API call failed: {api_err}")
            return _fallback_intent(query)

    except Exception as e:
        logger.error(f"Google GenAI SDK initialization failed: {e}")
        return _fallback_intent(query)

def _fallback_intent(query: str) -> QueryIntent:
    """Simple rule-based fallback."""
    logger.info("Using rule-based fallback for query interpretation")
    return QueryIntent(
        search_terms=query,
        genres=[],
        min_score=None,
        is_recommendation=True
    )
