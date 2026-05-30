from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
from app.core.vector_db import VectorDBInterface
import logging

logger = logging.getLogger(__name__)

class QdrantVectorDB(VectorDBInterface):
    def __init__(self, host: str = "localhost", port: int = 6333, path: Optional[str] = None, collection_name: str = "anime_embeddings", dimension: int = 768):
        if path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.dimension = dimension

    async def initialize(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
            )

    async def add_items(self, ids: List[int], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=metadata
            )
        )

    async def search(self, query_vector: List[float], top_n: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        qdrant_filter = None
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if value is not None:
                    if isinstance(value, list):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
            if filter_conditions:
                qdrant_filter = models.Filter(must=filter_conditions)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_n,
            query_filter=qdrant_filter
        ).points
        return [
            {
                "id": res.id,
                "score": res.score
            }
            for res in search_result
        ]

    async def get_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        res = self.client.retrieve(collection_name=self.collection_name, ids=[item_id])
        if res:
            return {"id": res[0].id, "entity": {**res[0].payload, "id": res[0].id}}
        return None

    async def count(self) -> int:
        return self.client.get_collection(self.collection_name).points_count
