from typing import List, Tuple, Dict, Any, Optional
import asyncio
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
from src.abstract_interface_classes import AbstractVectorDatabase, AbstractEmbeddingModel

class BaseMilvusVectorDatabase(AbstractVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", db_path: str = "./milvus.db"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.db_path = db_path
        self.dim = None
        self.collection = None

    async def initialize(self):
        self.dim = len(await self.embedding_model.embed("test sentence"))
        
        # Always use the local db file
        connections.connect("default", db_name=self.db_path)
        print(f"Connected to local database file: {self.db_path}")

        if not utility.has_collection(self.collection_name):
            await self._create_collection()
        
        self.collection = Collection(self.collection_name)
        self.collection.load()

    async def _create_collection(self):
        fields = [
            FieldSchema(name="media_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="description_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, "Media items collection")
        self.collection = Collection(self.collection_name, schema)

    async def add_items(self, ids: List[int], titles: List[str], descriptions: List[str]):
        title_embeddings = await self.embedding_model.embed_batch(titles)
        description_embeddings = await self.embedding_model.embed_batch(descriptions)
        self.collection.insert([ids, titles, title_embeddings, description_embeddings])
        self.collection.flush()

    async def find_similar_by_description(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = await self.embedding_model.embed(query)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="description_embedding",
            param=search_params,
            limit=k,
            output_fields=["media_id", "title"]
        )
        
        return [(hit.entity.get('media_id'), hit.entity.get('title'), hit.distance) for hit in results[0]]

    async def find_similar_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = await self.embedding_model.embed(title)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="title_embedding",
            param=search_params,
            limit=k + 1,
            output_fields=["media_id", "title"]
        )
        
        return [(hit.entity.get('media_id'), hit.entity.get('title'), hit.distance) for hit in results[0] if hit.entity.get('title') != title][:k]

    def close(self):
        if self.collection:
            self.collection.release()
        connections.disconnect("default")

class OptimizedMilvusVectorDatabase(BaseMilvusVectorDatabase):
    async def _create_collection(self):
        await super()._create_collection()
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 500}
        }
        self.collection.create_index("title_embedding", index_params)
        self.collection.create_index("description_embedding", index_params)

    async def find_similar_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = await self.embedding_model.embed(title)
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="title_embedding",
            param=search_params,
            limit=k + 1,
            output_fields=["media_id", "title"]
        )
        
        return [(hit.entity.get('media_id'), hit.entity.get('title'), hit.distance) for hit in results[0] if hit.entity.get('title') != title][:k]

    async def find_similar_by_description(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = await self.embedding_model.embed(query)
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="description_embedding",
            param=search_params,
            limit=k,
            output_fields=["media_id", "title"]
        )
        
        return [(hit.entity.get('media_id'), hit.entity.get('title'), hit.distance) for hit in results[0]]

class PersistentMilvusVectorDatabase(OptimizedMilvusVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", db_path: str = "./milvus.db", save_path: str = "./milvus_data"):
        super().__init__(embedding_model, collection_name, db_path)
        self.save_path = save_path

    def save(self):
        utility.save_collection(self.collection, self.save_path)

    def load(self):
        utility.load_collection(self.save_path)

    def update(self, media_id: int, title: str, description: str):
        title_embedding = self.embedding_model.embed(title).tolist()
        description_embedding = self.embedding_model.embed(description).tolist()
        
        self.collection.delete(expr=f"media_id == {media_id}")
        self.collection.insert([[media_id], [title], [title_embedding], [description_embedding]])
        self.collection.flush()

    def get(self, media_id: int) -> Optional[Dict[str, Any]]:
        results = self.collection.query(expr=f"media_id == {media_id}", output_fields=["media_id", "title"])
        if results:
            return {"media_id": results[0]["media_id"], "title": results[0]["title"]}
        return None

    def bulk_update(self, updates: List[Dict[str, Any]]):
        for update in updates:
            self.update(update['media_id'], update['title'], update['description'])

    def count(self) -> int:
        return self.collection.num_entities

    def clear(self):
        self.collection.drop()
        self._create_collection()
        self.collection = Collection(self.collection_name)
        self.collection.load()

async def create_vector_database(embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", db_path: str = "./milvus.db", save_path: str = "./milvus_data") -> PersistentMilvusVectorDatabase:
    db = PersistentMilvusVectorDatabase(embedding_model, collection_name, db_path, save_path)
    await db.initialize()
    return db
# Example usage:
# async def main():
#     db = await create_vector_database(embedding_model)
#     await db.add_items(ids, titles, descriptions)
#     results = await db.find_similar_by_description("A young boy becomes a powerful hero")
#     db.update(1, "Updated Media Title", "Updated description")
#     item = db.get(1)
#     count = db.count()
#     db.save()
#     db.clear()
#     db.close()
#
# asyncio.run(main())