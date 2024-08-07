from typing import List, Tuple
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
from abstract_classes import AbstractVectorDatabase, AbstractEmbeddingModel

class MilvusVectorDatabase(AbstractVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "anime_items"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.dim = len(self.embedding_model.embed("test sentence"))
        
        connections.connect("default", host="localhost", port="19530")
        
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
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index("title_embedding", index_params)
        self.collection.create_index("synopsis_embedding", index_params)

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