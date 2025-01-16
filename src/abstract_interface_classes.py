from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class AbstractEmbeddingModel(ABC):
    @abstractmethod
    async def get_text_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

class AbstractDataset(ABC):
    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

class AbstractVectorDatabase(ABC):
    @abstractmethod
    def add_items(self, ids: List[int], titles: List[str], synopses: List[str]):
        pass

    @abstractmethod
    def find_similar_by_description(self, query: str, k: int) -> List[Tuple[int, str, float]]:
        pass

    @abstractmethod
    def find_similar_by_title(self, title: str, k: int) -> List[Tuple[int, str, float]]:
        pass

class AbstractRecommendationEngine(ABC):
    def __init__(self, embedding_model: AbstractEmbeddingModel, vector_db: AbstractVectorDatabase, data_path: str, batch_size: int = 64):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.data_path = data_path
        self.batch_size = batch_size

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def get_recommendations_by_synopsis(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        pass

    @abstractmethod
    def get_recommendations_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        pass

class EmbeddingModelAdapter(AbstractEmbeddingModel):
    def __init__(self, model: AbstractEmbeddingModel):
        self.model = model

    async def get_text_embedding(self, text: str) -> List[float]:
        return await self.model.embed(text)

    async def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self.model.embed_batch(texts)