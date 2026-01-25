import torch
import numpy as np
import asyncio
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from pydantic import ConfigDict

class GemmaEmbeddingModel:
    """
    Modern embedding model using Google's EmbeddingGemma-300m.
    Optimized for performance and async execution using Sentence Transformers.
    """
    def __init__(self, model_name: str = "google/embeddinggemma-300m", device: str = None, token: Optional[str] = None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading embedding model: {model_name} on {self.device}")
        try:
            self.model = SentenceTransformer(
                model_name, 
                device=self.device,
                token=token
            )
        except Exception as e:
            print(f"Failed to load Gemma model: {e}")
            print("Falling back to Mock model for now. Please ensure you have accepted the model terms on Hugging Face.")
            self.model = None

    async def embed_text(self, text: str, task_type: str = "retrieval_query") -> List[float]:
        if self.model is None:
            return np.random.rand(768).tolist()
        return (await self.embed_batch([text], task_type=task_type))[0]

    async def embed_batch(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        if self.model is None:
            return [np.random.rand(768).tolist() for _ in texts]
        return await asyncio.to_thread(self._embed_sync, texts, task_type)

    def _embed_sync(self, texts: List[str], task_type: str) -> List[List[float]]:
        # Map task_type to prompts available in the model
        # Available keys: ['query', 'document', 'Retrieval-query', 'Retrieval-document', ...]
        prompt_name = None
        if task_type == "retrieval_query":
            prompt_name = "Retrieval-query"
        elif task_type == "retrieval_document":
            prompt_name = "Retrieval-document"
            
        # Using model.encode directly which handles pooling and normalization
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            prompt_name=prompt_name
        )
        return embeddings.tolist()

    async def encode(self, text: str) -> List[float]:
        """Alias for embed_text to maintain compatibility."""
        return await self.embed_text(text)

class MockEmbeddingModel:
    """Mock for testing or when GPU is not available."""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    async def embed_text(self, text: str) -> List[float]:
        return np.random.rand(self.dimension).tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [np.random.rand(self.dimension).tolist() for _ in texts]

    async def encode(self, text: str) -> List[float]:
        """Alias for embed_text to maintain compatibility."""
        return await self.embed_text(text)
