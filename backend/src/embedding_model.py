import torch
import numpy as np
import asyncio
from typing import List, Tuple, Union

# Keep imports in try-except blocks to prevent errors if dependencies are missing
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.abstract_interface_classes import AbstractEmbeddingModel

class OptimizedEmbeddingModel(AbstractEmbeddingModel):
    def __init__(self, model_name: str, use_sentence_transformers: bool = False, trust_remote_code: bool = False, device: str = None):
        self.model_name = model_name
        self.use_sentence_transformers = use_sentence_transformers
        self.trust_remote_code = trust_remote_code
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        if use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(model_name, device=self.device)
        elif TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            
            # Disable xformers if on CPU
            if self.device == 'cpu':
                self.model.config.use_xformers_for_memory_efficient_attention = False
            
            self.model.to(self.device)
        else:
            raise ImportError("Neither transformers nor sentence-transformers are available.")

    async def embed(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._embed_sync, [text])

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: List[str]) -> Union[np.ndarray, List[np.ndarray]]:
        try:
            if self.use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
            elif TRANSFORMERS_AVAILABLE:
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            else:
                raise RuntimeError("Embedding model is not properly initialized.")
            
            if len(texts) == 1:
                return embeddings.flatten()
            else:
                return [embedding.flatten() for embedding in embeddings]
        except Exception as e:
            print(f"Error during embedding: {str(e)}")
            raise

class QuantizedEmbeddingModel(OptimizedEmbeddingModel):
    def __init__(self, model_name: str, use_sentence_transformers: bool = False, trust_remote_code: bool = False, 
                    quantization_config: List[Tuple[str, int]] = [("dynamic", 8)], device: str = None):
        super().__init__(model_name, use_sentence_transformers, trust_remote_code, device)
        if not use_sentence_transformers and TRANSFORMERS_AVAILABLE:
            self.quantize_model(quantization_config)

    def quantize_model(self, quantization_config: List[Tuple[str, int]]):
        for q_type, q_bits in quantization_config:
            if q_type == "dynamic":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8 if q_bits == 8 else torch.float16
                )
            elif q_type == "static":
                self.model = torch.quantization.quantize_static(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8 if q_bits == 8 else torch.float16
                )
            elif q_type == "qat":
                self.model = torch.quantization.quantize_qat(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8 if q_bits == 8 else torch.float16
                )
            else:
                raise ValueError(f"Unsupported quantization type: {q_type}")

        self.model.to(self.device)

class MockEmbeddingModel(AbstractEmbeddingModel):
    """
    A simple mock embedding model for testing purposes that doesn't rely on external libraries.
    This avoids dependency issues with transformers and sentence-transformers.
    """
    def __init__(self, model_name: str = "mock-model", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        print(f"Initialized MockEmbeddingModel with dimension {dimension}")
        
    async def get_text_embedding(self, text: str) -> List[float]:
        """Generate a deterministic but unique mock embedding for a text"""
        # Generate a deterministic embedding based on the text
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
        # Normalize the embedding to unit length
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
        
    async def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for a list of texts"""
        results = []
        for text in texts:
            embedding = await self.get_text_embedding(text)
            results.append(embedding)
        return results

class SentenceTransformerEmbeddingModel(AbstractEmbeddingModel):
    """
    Implementation of the AbstractEmbeddingModel interface using sentence-transformers.
    This class is used by the tests and production code for the recommendation engine.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)

    async def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        return await asyncio.to_thread(self._get_embedding, text)

    async def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        return await asyncio.to_thread(self._get_embeddings, texts)

    def _get_embedding(self, text: str) -> List[float]:
        """Synchronous method to get a single embedding"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Synchronous method to get multiple embeddings"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]

# Example usage:
# model = SentenceTransformerEmbeddingModel("bert-base-uncased")
# async def get_embedding(text):
#     return await model.get_text_embedding(text)