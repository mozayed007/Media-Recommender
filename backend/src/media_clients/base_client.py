from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import aiohttp
import logging

logger = logging.getLogger(__name__)

class MediaClient(ABC):
    """Abstract base class for all media API clients."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def search(self, query: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Search for media by query string.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of raw media data dictionaries
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, media_id: str) -> Dict[str, Any]:
        """Get media details by ID.
        
        Args:
            media_id: Unique identifier for the media
            
        Returns:
            Raw media data dictionary
        """
        pass
    
    @abstractmethod
    async def get_popular(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get popular/trending media.
        
        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of raw media data dictionaries
        """
        pass
    
    @abstractmethod
    def normalize_to_media_base(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw API response to MediaBase-compatible dictionary.
        
        Args:
            raw_data: Raw data from API
            
        Returns:
            Dictionary compatible with MediaBase model
        """
        pass
    
    async def initialize(self):
        """Initialize HTTP session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.logger.info(f"Initialized HTTP session for {self.__class__.__name__}")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info(f"Closed HTTP session for {self.__class__.__name__}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
