from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorDBInterface(ABC):
    @abstractmethod
    async def initialize(self):
        """Initialize the database connection and collection."""
        pass

    @abstractmethod
    async def add_items(self, ids: List[int], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """Add items with embeddings and metadata to the database."""
        pass

    @abstractmethod
    async def search(self, query_vector: List[float], top_n: int = 10, filter_expr: str = "") -> List[Dict[str, Any]]:
        """Search for similar items in the database."""
        pass

    @abstractmethod
    async def get_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve an item by its ID."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get the total number of items in the collection."""
        pass
