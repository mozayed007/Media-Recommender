import os
import asyncio
import logging
from typing import List, Tuple, Dict, Any, Optional
from pymilvus import MilvusClient, DataType
from app.core.vector_db import VectorDBInterface

logger = logging.getLogger(__name__)

class MilvusVectorDB(VectorDBInterface):
    def __init__(
        self, 
        uri: str = "http://localhost:19530", 
        collection_name: str = "anime_embeddings",
        dimension: int = 768
    ):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.dimension = dimension

    async def initialize(self):
        if self.client.has_collection(self.collection_name):
            logger.info(f"Collection {self.collection_name} already exists.")
            return

        logger.info(f"Creating collection {self.collection_name}...")
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            auto_id=False,
            enable_dynamic_field=True
        )
        logger.info(f"Collection {self.collection_name} created.")

    async def add_items(self, ids: List[int], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        data = []
        for i in range(len(ids)):
            item = {
                "id": ids[i],
                "vector": embeddings[i],
                **metadata[i]
            }
            data.append(item)
        
        self.client.insert(collection_name=self.collection_name, data=data)

    async def search(self, query_vector: List[float], top_n: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ALLOWED_FILTER_KEYS = {"media_type", "genres", "score", "status"}
        filter_expr = ""
        if filters:
            expressions = []
            for key, value in filters.items():
                if key not in ALLOWED_FILTER_KEYS:
                    continue
                if isinstance(value, str):
                    sanitized = value.replace('"', '').replace("'", "")
                    expressions.append(f'{key} == "{sanitized}"')
                elif isinstance(value, (int, float)):
                    expressions.append(f'{key} == {value}')
            filter_expr = " and ".join(expressions)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_n,
            filter=filter_expr,
            output_fields=["*"]
        )
        # Standardize output to match RecommenderService expectation
        return [
            {
                "id": hit["id"],
                "score": hit["distance"]  # Milvus returns distance
            }
            for hit in results[0]
        ]

    async def get_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        res = self.client.get(collection_name=self.collection_name, ids=[item_id])
        return res[0] if res else None

    async def count(self) -> int:
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return stats.get("row_count", 0)
