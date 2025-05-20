from abc import ABC, abstractmethod
from typing import List, Tuple

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
        
    @abstractmethod
    def save(self):
        """Save any pending changes to the vector database."""
        pass
        
    @abstractmethod
    def count(self) -> int:
        """Return the count of items in the vector database.
        
        Returns:
            int: Number of items in the database.
        """
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
    async def get_recommendations_by_description(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        pass

    @abstractmethod
    async def get_recommendations_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        pass

class EmbeddingModelAdapter(AbstractEmbeddingModel):
    def __init__(self, model):
        """Adapter for models that don't directly implement AbstractEmbeddingModel.
        
        Args:
            model: A model object that has embed/embed_batch or similar methods.
        """
        self.model = model

    async def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using the underlying model.
        
        Args:
            text: The text to embed.
            
        Returns:
            List[float]: The embedding vector.
        """
        # Try different method names that could exist on the adapted model
        if hasattr(self.model, 'get_text_embedding'):
            return await self.model.get_text_embedding(text)
        elif hasattr(self.model, 'embed'):
            return await self.model.embed(text)
        else:
            raise AttributeError("Underlying model has no compatible embedding method")

    async def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using the underlying model.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        # Try different method names that could exist on the adapted model
        if hasattr(self.model, 'get_text_embeddings'):
            return await self.model.get_text_embeddings(texts)
        elif hasattr(self.model, 'embed_batch'):
            return await self.model.embed_batch(texts)
        else:
            # Fall back to single embeddings if batch method not available
            return [await self.get_text_embedding(text) for text in texts]