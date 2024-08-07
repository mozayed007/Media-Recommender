from typing import List, Tuple, Dict, Any, Optional
import asyncio
import uuid
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
from abstract_classes import AbstractVectorDatabase, AbstractEmbeddingModel

class BaseMilvusVectorDatabase(AbstractVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "anime_items", host: str = "localhost", port: str = "19530"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.dim = len(self.embedding_model.embed("test sentence"))
        self.host = host
        self.port = port
        
        connections.connect("default", host=self.host, port=self.port)
        
        if not utility.has_collection(self.collection_name):
            self._create_collection()
        
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _create_collection(self):
        fields = [
            FieldSchema(name="anime_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="synopsis_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, "Anime items collection")
        self.collection = Collection(self.collection_name, schema)

    def add_items(self, ids: List[int], titles: List[str], synopses: List[str]):
        title_embeddings = [self.embedding_model.embed(title).tolist() for title in titles]
        synopsis_embeddings = [self.embedding_model.embed(synopsis).tolist() for synopsis in synopses]
        self.collection.insert([ids, titles, title_embeddings, synopsis_embeddings])

    def find_similar_by_synopsis(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = self.embedding_model.embed(query)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="synopsis_embedding",
            param=search_params,
            limit=k,
            output_fields=["anime_id", "title"]
        )
        
        return [(hit.entity.get('anime_id'), hit.entity.get('title'), hit.distance) for hit in results[0]]

    def find_similar_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = self.embedding_model.embed(title)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="title_embedding",
            param=search_params,
            limit=k + 1,
            output_fields=["anime_id", "title"]
        )
        
        return [(hit.entity.get('anime_id'), hit.entity.get('title'), hit.distance) for hit in results[0] if hit.entity.get('title') != title][:k]

    def close(self):
        connections.disconnect("default")

class OptimizedMilvusVectorDatabase(BaseMilvusVectorDatabase):
    def _create_collection(self):
        super()._create_collection()
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 500}
        }
        self.collection.create_index("title_embedding", index_params)
        self.collection.create_index("synopsis_embedding", index_params)

    def find_similar_by_synopsis(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = self.embedding_model.embed(query)
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="synopsis_embedding",
            param=search_params,
            limit=k,
            output_fields=["anime_id", "title"]
        )
        
        return [(hit.entity.get('anime_id'), hit.entity.get('title'), hit.distance) for hit in results[0]]

    def find_similar_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = self.embedding_model.embed(title)
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="title_embedding",
            param=search_params,
            limit=k + 1,
            output_fields=["anime_id", "title"]
        )
        
        return [(hit.entity.get('anime_id'), hit.entity.get('title'), hit.distance) for hit in results[0] if hit.entity.get('title') != title][:k]

class ShardedMilvusVectorDatabase(OptimizedMilvusVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "anime_items", hosts: List[str] = ["localhost"], ports: List[str] = ["19530"]):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.dim = len(self.embedding_model.embed("test sentence"))
        self.num_shards = len(hosts)
        
        self.shards = []
        for i, (host, port) in enumerate(zip(hosts, ports)):
            connections.connect(f"shard_{i}", host=host, port=port)
            if not utility.has_collection(f"{self.collection_name}_{i}"):
                self._create_collection(i)
            collection = Collection(f"{self.collection_name}_{i}")
            collection.load()
            self.shards.append(collection)

    def _create_collection(self, shard_id: int):
        fields = [
            FieldSchema(name="anime_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="synopsis_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, f"Anime items collection shard {shard_id}")
        collection = Collection(f"{self.collection_name}_{shard_id}", schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 500}
        }
        collection.create_index("title_embedding", index_params)
        collection.create_index("synopsis_embedding", index_params)

    async def add_items_batch(self, ids: List[int], titles: List[str], synopses: List[str], batch_size: int = 1000):
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_titles = titles[i:i+batch_size]
            batch_synopses = synopses[i:i+batch_size]
            
            title_embeddings = await self.embedding_model.embed_batch(batch_titles)
            synopsis_embeddings = await self.embedding_model.embed_batch(batch_synopses)
            
            for j, anime_id in enumerate(batch_ids):
                shard_id = anime_id % self.num_shards
                self.shards[shard_id].insert([[anime_id], [batch_titles[j]], [title_embeddings[j]], [synopsis_embeddings[j]]])
        
        for shard in self.shards:
            shard.flush()

    async def find_similar_by_synopsis(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_embedding = await self.embedding_model.embed(query)
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        
        results = await asyncio.gather(*[
            self._search_shard(shard, query_embedding, "synopsis_embedding", search_params, k)
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
            output_fields=["anime_id", "title"]
        )
        return [(hit.entity.get('anime_id'), hit.entity.get('title'), hit.distance) for hit in results[0]]

    def close(self):
        for i in range(self.num_shards):
            connections.disconnect(f"shard_{i}")

class PersistentShardedMilvusVectorDatabase(ShardedMilvusVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "anime_items", hosts: List[str] = ["localhost"], ports: List[str] = ["19530"], save_path: str = "./milvus_data"):
        super().__init__(embedding_model, collection_name, hosts, ports)
        self.save_path = save_path

    def save(self):
        for i, shard in enumerate(self.shards):
            utility.save_collection(shard, f"{self.save_path}/shard_{i}")

    def load(self):
        for i, shard in enumerate(self.shards):
            utility.load_collection(f"{self.save_path}/shard_{i}")

    def update(self, anime_id: int, title: str, synopsis: str):
        shard_id = anime_id % self.num_shards
        shard = self.shards[shard_id]
        
        title_embedding = self.embedding_model.embed(title).tolist()
        synopsis_embedding = self.embedding_model.embed(synopsis).tolist()
        
        shard.delete(expr=f"anime_id == {anime_id}")
        shard.insert([[anime_id], [title], [title_embedding], [synopsis_embedding]])

    def get(self, anime_id: int) -> Optional[Dict[str, Any]]:
        shard_id = anime_id % self.num_shards
        shard = self.shards[shard_id]
        
        results = shard.query(expr=f"anime_id == {anime_id}", output_fields=["anime_id", "title"])
        if results:
            return {"anime_id": results[0]["anime_id"], "title": results[0]["title"]}
        return None

    def bulk_update(self, updates: List[Dict[str, Any]]):
        for update in updates:
            self.update(update['anime_id'], update['title'], update['synopsis'])

    def count(self) -> int:
        return sum(shard.num_entities for shard in self.shards)

    def clear(self):
        for i, shard in enumerate(self.shards):
            shard.drop()
            self._create_collection(i)
            shard = Collection(f"{self.collection_name}_{i}")
            shard.load()
            self.shards[i] = shard

# Example usage:
# db = PersistentShardedMilvusVectorDatabase(embedding_model, "anime_items", 
#                                            hosts=['localhost', 'localhost'], 
#                                            ports=['19530', '19531'], 
#                                            save_path="./milvus_data")
# await db.add_items_batch(ids, titles, synopses)
# results = await db.find_similar_by_synopsis("A young boy becomes a powerful ninja")
# db.save()
# db.update(1, "Updated Anime Title", "Updated synopsis")
# item = db.get(1)
# count = db.count()
# db.clear()
# db.close()