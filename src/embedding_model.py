import torch
import numpy as np
import asyncio
from typing import List, Tuple, Union
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from src.abstract_interface_classes import AbstractEmbeddingModel

class OptimizedEmbeddingModel(AbstractEmbeddingModel):
    def __init__(self, model_name: str, use_sentence_transformers: bool = False, trust_remote_code: bool = False, device: str = None):
        self.model_name = model_name
        self.use_sentence_transformers = use_sentence_transformers
        self.trust_remote_code = trust_remote_code
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        if use_sentence_transformers:
            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            
            # Disable xformers if on CPU
            if self.device == 'cpu':
                self.model.config.use_xformers_for_memory_efficient_attention = False
            
            self.model.to(self.device)

    async def embed(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._embed_sync, [text])

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: List[str]) -> Union[np.ndarray, List[np.ndarray]]:
        try:
            if self.use_sentence_transformers:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
            else:
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
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
        if not use_sentence_transformers:
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

# Example usage:
# model = QuantizedEmbeddingModel("bert-base-uncased", 
#     quantization_config=[("dynamic", 8)], device="cuda")
# async def get_embedding(text):
#     return await model.embed(text)