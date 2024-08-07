from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
from transformers import AutoModel, AutoTokenizer

class AbstractEmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

class HuggingFaceEmbeddingModel(AbstractEmbeddingModel):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings[0]

class AbstractDataset(Dataset):
    @abstractmethod
    def __getitem__(self, idx):
        pass

class AnimeDataset(AbstractDataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return item['anime_id'], item['title'], item['synopsis']

class AbstractVectorDatabase(ABC):
    @abstractmethod
    def add_items(self, ids: List[int], titles: List[str], synopses: List[str]):
        pass

    @abstractmethod
    def find_similar_by_synopsis(self, query: str, k: int) -> List[Tuple[int, str, float]]:
        pass

    @abstractmethod
    def find_similar_by_title(self, title: str, k: int) -> List[Tuple[int, str, float]]:
        pass

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

class AnimeRecommendationEngine:
    def __init__(self, embedding_model: AbstractEmbeddingModel, vector_db: AbstractVectorDatabase, data_path: str, batch_size: int = 64):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.dataset = AnimeDataset(data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def load_data(self):
        for batch in self.dataloader:
            ids, titles, synopses = batch
            self.vector_db.add_items(ids, titles, synopses)

    def get_recommendations_by_synopsis(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        return self.vector_db.find_similar_by_synopsis(query, k)

    def get_recommendations_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        return self.vector_db.find_similar_by_title(title, k)

# Example usage:
model_name = "dunzhang/stella_en_400M_v5"  # or "dunzhang/stella_en_1.5B_v5"
embedding_model = HuggingFaceEmbeddingModel(model_name)
vector_db = MilvusVectorDatabase(embedding_model)
data_path = "../data/processed/anime_mal_August.csv"  # Update this path to your processed data
rec_engine = AnimeRecommendationEngine(embedding_model, vector_db, data_path)
rec_engine.load_data()

# Recommendation by synopsis
query_synopsis = "A thrilling space adventure with giant robots"
recommendations_by_synopsis = rec_engine.get_recommendations_by_synopsis(query_synopsis, k=5)
print("Recommendations based on synopsis:")
for anime_id, title, distance in recommendations_by_synopsis:
    print(f"{anime_id} - {title}: {distance}")

# Recommendation by title
query_title = "Neon Genesis Evangelion"
recommendations_by_title = rec_engine.get_recommendations_by_title(query_title, k=5)
print("\nRecommendations based on title:")
for anime_id, title, distance in recommendations_by_title:
    print(f"{anime_id} - {title}: {distance}")