"""Error handler for ingestion pipeline.

Provides error categorization, retry logic, and logging for ingestion errors.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of ingestion errors."""
    TRANSIENT = "transient"  # Network errors, timeouts - should retry
    RATE_LIMIT = "rate_limit"  # 429 responses - backoff and retry
    PERMANENT = "permanent"  # 404, 401 - don't retry
    DATA_QUALITY = "data_quality"  # Invalid data format
    UNKNOWN = "unknown"  # Unexpected errors


class IngestionErrorHandler:
    """Handle errors during data ingestion with categorization and retry logic."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 60.0,
    ):
        """Initialize error handler.
        
        Args:
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff multiplier
            max_backoff_seconds: Maximum backoff time in seconds
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_stats: Dict[str, Dict[str, Any]] = {}
        self.error_log: list = []
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error for appropriate handling.
        
        Args:
            error: The exception to categorize
            
        Returns:
            ErrorCategory enum value
        """
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Rate limit errors
        if "rate limit" in error_message or "429" in error_message:
            return ErrorCategory.RATE_LIMIT
        
        # Transient network errors
        transient_keywords = [
            "timeout", "connection", "network", "temporary",
            "unavailable", "503", "502", "504"
        ]
        if any(keyword in error_message for keyword in transient_keywords):
            return ErrorCategory.TRANSIENT
        
        # Permanent errors
        permanent_keywords = [
            "not found", "404", "unauthorized", "401", "forbidden", "403",
            "invalid api key", "invalid client"
        ]
        if any(keyword in error_message for keyword in permanent_keywords):
            return ErrorCategory.PERMANENT
        
        # Data quality errors
        data_keywords = [
            "validation", "parse", "decode", "json", "format"
        ]
        if any(keyword in error_message for keyword in data_keywords):
            return ErrorCategory.DATA_QUALITY
        
        return ErrorCategory.UNKNOWN
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should be retried.
        
        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-indexed)
            
        Returns:
            True if should retry, False otherwise
        """
        if attempt >= self.max_retries:
            return False
        
        category = self.categorize_error(error)
        
        # Don't retry permanent errors
        if category == ErrorCategory.PERMANENT:
            return False
        
        # Don't retry data quality errors
        if category == ErrorCategory.DATA_QUALITY:
            return False
        
        # Retry transient and rate limit errors
        return category in [ErrorCategory.TRANSIENT, ErrorCategory.RATE_LIMIT]
    
    def get_backoff_time(self, attempt: int, error: Optional[Exception] = None) -> float:
        """Calculate backoff time for retry.
        
        Args:
            attempt: Attempt number (0-indexed)
            error: Optional error for context
            
        Returns:
            Backoff time in seconds
        """
        # Base backoff with exponential increase
        backoff = self.backoff_factor ** attempt
        
        # Add extra delay for rate limits
        if error and self.categorize_error(error) == ErrorCategory.RATE_LIMIT:
            backoff *= 2
        
        return min(backoff, self.max_backoff_seconds)
    
    async def execute_with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with automatic retry logic.
        
        Args:
            operation: Async callable to execute
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Operation result
            
        Raises:
            Exception: If all retries exhausted
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                return await operation(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                category = self.categorize_error(e)
                
                if not self.should_retry(e, attempt):
                    self.logger.error(
                        f"Operation failed with {category.value} error, not retrying: {e}"
                    )
                    raise
                
                backoff = self.get_backoff_time(attempt, e)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed ({category.value}): {e}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                
                await asyncio.sleep(backoff)
                attempt += 1
        
        # All retries exhausted
        raise last_error
    
    def log_error(
        self, 
        source: str, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ):
        """Log an error with context.
        
        Args:
            source: Data source identifier
            error: The exception
            context: Additional context (endpoint, params, etc.)
        """
        category = self.categorize_error(error)
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "category": category.value,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }
        
        self.error_log.append(error_entry)
        
        # Update stats
        if source not in self.error_stats:
            self.error_stats[source] = {
                "total": 0,
                "by_category": {},
                "by_type": {},
            }
        
        self.error_stats[source]["total"] += 1
        
        # Count by category
        cat_key = category.value
        self.error_stats[source]["by_category"][cat_key] = \
            self.error_stats[source]["by_category"].get(cat_key, 0) + 1
        
        # Count by error type
        type_key = type(error).__name__
        self.error_stats[source]["by_type"][type_key] = \
            self.error_stats[source]["by_type"].get(type_key, 0) + 1
        
        # Log with appropriate level
        if category == ErrorCategory.PERMANENT:
            self.logger.error(f"[{source}] {type(error).__name__}: {error}")
        elif category == ErrorCategory.RATE_LIMIT:
            self.logger.warning(f"[{source}] Rate limit hit")
        else:
            self.logger.debug(f"[{source}] {type(error).__name__}: {error}")
    
    def get_error_stats(self, source: Optional[str] = None) -> Dict[str, Any]:
        """Get error statistics.
        
        Args:
            source: Optional source to filter by
            
        Returns:
            Dictionary of error statistics
        """
        if source:
            return self.error_stats.get(source, {})
        
        # Aggregate all sources
        total_stats = {
            "total_errors": sum(s["total"] for s in self.error_stats.values()),
            "by_source": {k: v["total"] for k, v in self.error_stats.items()},
            "by_category": {},
            "sources": list(self.error_stats.keys()),
        }
        
        # Aggregate by category across all sources
        for stats in self.error_stats.values():
            for cat, count in stats.get("by_category", {}).items():
                total_stats["by_category"][cat] = \
                    total_stats["by_category"].get(cat, 0) + count
        
        return total_stats
    
    def get_recent_errors(
        self, 
        source: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        limit: int = 10
    ) -> list:
        """Get recent error entries.
        
        Args:
            source: Filter by source
            category: Filter by category
            limit: Maximum entries to return
            
        Returns:
            List of recent error entries
        """
        filtered = self.error_log
        
        if source:
            filtered = [e for e in filtered if e["source"] == source]
        
        if category:
            filtered = [e for e in filtered if e["category"] == category.value]
        
        return filtered[-limit:]
    
    def reset_stats(self, source: Optional[str] = None):
        """Reset error statistics.
        
        Args:
            source: Specific source to reset, or None for all
        """
        if source:
            if source in self.error_stats:
                del self.error_stats[source]
        else:
            self.error_stats.clear()
            self.error_log.clear()
    
    def has_critical_errors(self, source: str, threshold: int = 10) -> bool:
        """Check if source has exceeded error threshold.
        
        Args:
            source: Source identifier
            threshold: Error count threshold
            
        Returns:
            True if critical error level reached
        """
        stats = self.error_stats.get(source, {})
        return stats.get("total", 0) >= threshold
    
    def get_error_summary(self) -> str:
        """Get human-readable error summary.
        
        Returns:
            Formatted error summary string
        """
        stats = self.get_error_stats()
        
        if stats["total_errors"] == 0:
            return "No errors recorded."
        
        lines = [
            f"Error Summary ({stats['total_errors']} total):",
            "By Source:",
        ]
        
        for source, count in sorted(stats["by_source"].items()):
            lines.append(f"  {source}: {count}")
        
        lines.append("By Category:")
        for cat, count in sorted(stats["by_category"].items()):
            lines.append(f"  {cat}: {count}")
        
        return "\n".join(lines)
