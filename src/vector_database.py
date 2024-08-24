import os
from typing import List, Tuple, Dict, Any, Optional
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
from src.abstract_interface_classes import AbstractVectorDatabase, AbstractEmbeddingModel

class OptimizedMilvusLiteVectorDatabase(AbstractVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", db_path: str = "./milvus_lite_data"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.db_path = db_path
        self.dim = None
        self.collection = None

    async def initialize(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.dim = len(await self.embedding_model.embed("test sentence"))
        
        connections.connect("default", uri=self.db_path)
        print(f"Connected to Milvus Lite database at: {self.db_path}")

        if not utility.has_collection(self.collection_name):
            self._create_collection()
        else:
            self.collection = Collection(self.collection_name)
        
        if self.collection is not None:
            self.collection.load()
        else:
            raise ValueError(f"Failed to create or load collection: {self.collection_name}")

    def _create_collection(self):
        fields = [
            FieldSchema(name="media_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="description_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, "Media items collection")
        self.collection = Collection(self.collection_name, schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 500}
        }
        self.collection.create_index("title_embedding", index_params)
        self.collection.create_index("description_embedding", index_params)

    async def add_items(self, ids: List[int], titles: List[str], descriptions: List[str]):
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        title_embeddings = await self.embedding_model.embed_batch(titles)
        description_embeddings = await self.embedding_model.embed_batch(descriptions)
        entities = [ids, titles, title_embeddings, description_embeddings]
        await self.collection.insert(entities)
        await self.collection.flush()

    async def add_items_batch(self, ids: List[int], titles: List[str], descriptions: List[str], batch_size: int = 1000):
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_titles = titles[i:i+batch_size]
            batch_descriptions = descriptions[i:i+batch_size]
            await self.add_items(batch_ids, batch_titles, batch_descriptions)

    async def find_similar_by_description(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        query_embedding = await self.embedding_model.embed(query)
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        results = await self.collection.search(
            data=[query_embedding],
            anns_field="description_embedding",
            param=search_params,
            limit=k,
            output_fields=["media_id", "title"]
        )
        
        return [(hit.entity.get('media_id'), hit.entity.get('title'), hit.distance) for hit in results[0]]

    async def find_similar_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        query_embedding = await self.embedding_model.embed(title)
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        results = await self.collection.search(
            data=[query_embedding],
            anns_field="title_embedding",
            param=search_params,
            limit=k + 1,
            output_fields=["media_id", "title"]
        )
        
        return [(hit.entity.get('media_id'), hit.entity.get('title'), hit.distance) for hit in results[0] if hit.entity.get('title') != title][:k]

    async def update(self, media_id: int, title: str, description: str):
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        title_embedding = await self.embedding_model.embed(title)
        description_embedding = await self.embedding_model.embed(description)
        
        await self.collection.delete(expr=f"media_id == {media_id}")
        await self.collection.insert([[media_id], [title], [title_embedding], [description_embedding]])
        await self.collection.flush()

    async def get(self, media_id: int) -> Optional[Dict[str, Any]]:
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        results = await self.collection.query(expr=f"media_id == {media_id}", output_fields=["media_id", "title"])
        if results:
            return {"media_id": results[0]["media_id"], "title": results[0]["title"]}
        return None

    async def count(self) -> int:
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        return await self.collection.num_entities

    async def clear(self):
        if self.collection is None:
            raise ValueError("Collection is not initialized")
        await self.collection.drop()
        await self._create_collection()
        self.collection = Collection(self.collection_name)
        await self.collection.load()

    def save(self):
        # Milvus Lite automatically persists data, so we don't need to do anything here
        pass

    async def close(self):
        if self.collection:
            await self.collection.release()
        await connections.disconnect("default")

async def create_vector_database(embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", db_path: str = "./milvus_lite_data.db") -> OptimizedMilvusLiteVectorDatabase:
    db = OptimizedMilvusLiteVectorDatabase(embedding_model, collection_name, db_path)
    await db.initialize()
    return db