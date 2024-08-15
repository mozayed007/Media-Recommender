from typing import List, Tuple, Dict, Any, Optional
import asyncio
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility,
    MilvusClient
)
from src.abstract_interface_classes import AbstractVectorDatabase, AbstractEmbeddingModel

class BaseMilvusVectorDatabase(AbstractVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", host: str = "localhost", port: str = "19530"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dim = None
        self.collection = None
        self.client = MilvusClient("/data/processed/vector_db/milvus_demo.db")

    async def initialize(self):
        self.dim = len(await self.embedding_model.embed("test sentence"))
        try:
            connections.connect("default", host=self.host, port=self.port)
        except:
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
        connections.disconnect("default")

class OptimizedMilvusVectorDatabase(BaseMilvusVectorDatabase):
    async def _create_collection(self):
        fields = [
            FieldSchema(name="media_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="description_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, "Media items collection")
        self.collection = Collection(self.collection_name, schema)

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
class ShardedMilvusVectorDatabase(OptimizedMilvusVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", hosts: List[str] = ["localhost"], ports: List[str] = ["19530"]):
        super().__init__(embedding_model, collection_name)
        self.hosts = hosts
        self.ports = ports
        self.num_shards = len(hosts)
        self.shards = []

    async def initialize(self):
        self.dim = len(await self.embedding_model.embed("test sentence"))
        for i, (host, port) in enumerate(zip(self.hosts, self.ports)):
            connections.connect(f"shard_{i}", host=host, port=port)
            if not utility.has_collection(f"{self.collection_name}_{i}"):
                await self._create_collection(i)
            collection = Collection(f"{self.collection_name}_{i}")
            collection.load()
            self.shards.append(collection)

    async def _create_collection(self, shard_id: int):
        fields = [
            FieldSchema(name="media_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="description_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, f"Media items collection shard {shard_id}")
        collection = Collection(f"{self.collection_name}_{shard_id}", schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 500}
        }
        collection.create_index("title_embedding", index_params)
        collection.create_index("description_embedding", index_params)

    async def add_items_batch(self, ids: List[int], titles: List[str], descriptions: List[str], batch_size: int = 1000):
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_titles = titles[i:i+batch_size]
            batch_descriptions = descriptions[i:i+batch_size]
            
            title_embeddings = await self.embedding_model.embed_batch(batch_titles)
            description_embeddings = await self.embedding_model.embed_batch(batch_descriptions)
            
            for j, media_id in enumerate(batch_ids):
                shard_id = media_id % self.num_shards
                self.shards[shard_id].insert([[media_id], [batch_titles[j]], [title_embeddings[j]], [description_embeddings[j]]])
        
        for shard in self.shards:
            shard.flush()

    async def find_similar_by_description(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = await self.embedding_model.embed(query)
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        
        results = await asyncio.gather(*[
            self._search_shard(shard, query_embedding, "description_embedding", search_params, k)
            for shard in self.shards
        ])
        
        combined_results = [item for sublist in results for item in sublist]
        combined_results.sort(key=lambda x: x[2])  # Sort by distance
        return combined_results[:k]

    async def find_similar_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = await self.embedding_model.embed(title)
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        
        results = await asyncio.gather(*[
            self._search_shard(shard, query_embedding, "title_embedding", search_params, k + 1)
            for shard in self.shards
        ])
        
        combined_results = [item for sublist in results for item in sublist if item[1] != title]
        combined_results.sort(key=lambda x: x[2])  # Sort by distance
        return combined_results[:k]

    async def _search_shard(self, shard, query_embedding, field, search_params, k):
        results = shard.search(
            data=[query_embedding.tolist()],
            anns_field=field,
            param=search_params,
            limit=k,
            output_fields=["media_id", "title"]
        )
        return [(hit.entity.get('media_id'), hit.entity.get('title'), hit.distance) for hit in results[0]]

    def close(self):
        for i in range(self.num_shards):
            connections.disconnect(f"shard_{i}")

class PersistentShardedMilvusVectorDatabase(ShardedMilvusVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", hosts: List[str] = ["localhost"], ports: List[str] = ["19530"], save_path: str = "./milvus_data"):
        super().__init__(embedding_model, collection_name, hosts, ports)
        self.save_path = save_path

    def save(self):
        for i, shard in enumerate(self.shards):
            utility.save_collection(shard, f"{self.save_path}/shard_{i}")

    def load(self):
        for i, shard in enumerate(self.shards):
            utility.load_collection(f"{self.save_path}/shard_{i}")

    def update(self, media_id: int, title: str, description: str):
        shard_id = media_id % self.num_shards
        shard = self.shards[shard_id]
        
        title_embedding = self.embedding_model.embed(title).tolist()
        description_embedding = self.embedding_model.embed(description).tolist()
        
        shard.delete(expr=f"media_id == {media_id}")
        shard.insert([[media_id], [title], [title_embedding], [description_embedding]])

    def get(self, media_id: int) -> Optional[Dict[str, Any]]:
        shard_id = media_id % self.num_shards
        shard = self.shards[shard_id]
        
        results = shard.query(expr=f"media_id == {media_id}", output_fields=["media_id", "title"])
        if results:
            return {"media_id": results[0]["media_id"], "title": results[0]["title"]}
        return None

    def bulk_update(self, updates: List[Dict[str, Any]]):
        for update in updates:
            self.update(update['media_id'], update['title'], update['description'])

    def count(self) -> int:
        return sum(shard.num_entities for shard in self.shards)

    def clear(self):
        for i, shard in enumerate(self.shards):
            shard.drop()
            self._create_collection(i)
            shard = Collection(f"{self.collection_name}_{i}")
            shard.load()
            self.shards[i] = shard

async def create_vector_database(embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", hosts: List[str] = ["localhost"], ports: List[str] = ["19530"], save_path: str = "./milvus_data") -> PersistentShardedMilvusVectorDatabase:
    db = PersistentShardedMilvusVectorDatabase(embedding_model, collection_name, hosts, ports, save_path)
    await db.initialize()
    return db

# Example usage:
# db = PersistentShardedMilvusVectorDatabase(embedding_model, "media_items", 
#                                            hosts=['localhost', 'localhost'], 
#                                            ports=['19530', '19531'], 
#                                            save_path="./milvus_data")
# await db.add_items_batch(ids, titles, descriptions)
# results = await db.find_similar_by_description("A young boy becomes a powerful hero")
# db.save()
# db.update(1, "Updated Media Title", "Updated description")
# item = db.get(1)
# count = db.count()
# db.clear()
# db.close()