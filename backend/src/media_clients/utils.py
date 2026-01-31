import asyncio
import time
from typing import Callable, Any
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, rate: int, per: float):
        """Initialize rate limiter.
        
        Args:
            rate: Number of requests allowed
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = float(rate)
        self.last_check = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until a request can be made."""
        async with self._lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = float(self.rate)
            
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0

def with_retry(max_attempts: int = 3):
    """Decorator for retrying failed API calls with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, asyncio.TimeoutError)),
        reraise=True
    )

class APIError(Exception):
    """Base exception for API errors."""
    pass

class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    pass

class NotFoundError(APIError):
    """Raised when resource is not found."""
    pass

class ValidationError(APIError):
    """Raised when data validation fails."""
    pass
