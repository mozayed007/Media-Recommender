import torch
import numpy as np
from typing import List, Union, Tuple
from transformers import AutoModel, AutoTokenizer
from abstract_classes import AbstractEmbeddingModel

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

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.tolist()

class AsyncHuggingFaceEmbeddingModel(HuggingFaceEmbeddingModel):
    async def embed(self, text: str) -> np.ndarray:
        return await torch.cuda.current_stream().run_coroutine(super().embed(text))

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return await torch.cuda.current_stream().run_coroutine(super().embed_batch(texts))

class QuantizedHuggingFaceEmbeddingModel(AsyncHuggingFaceEmbeddingModel):
    def __init__(self, model_name: str, quantization_config: List[Tuple[str, int]] = [("dynamic", 8)]):
        super().__init__(model_name)
        self.quantize_model(quantization_config)

    def quantize_model(self, quantization_config: List[Tuple[str, int]]):
        for q_type, q_bits in quantization_config:
            if q_type == "dynamic":
                if q_bits == 8:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                elif q_bits == 4:
                    # Note: 4-bit dynamic quantization is not directly supported in PyTorch
                    # This is a placeholder for when it becomes available
                    print("4-bit dynamic quantization is not currently supported")
                elif q_bits == 16:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.float16
                    )
            elif q_type == "static":
                if q_bits == 8:
                    self.model = torch.quantization.quantize_static(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                elif q_bits == 4:
                    # Note: 4-bit static quantization is not directly supported in PyTorch
                    # This is a placeholder for when it becomes available
                    print("4-bit static quantization is not currently supported")
                elif q_bits == 16:
                    self.model = torch.quantization.quantize_static(
                        self.model, {torch.nn.Linear}, dtype=torch.float16
                    )
            elif q_type == "qat":
                if q_bits == 8:
                    self.model = torch.quantization.quantize_qat(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                elif q_bits == 4:
                    # Note: 4-bit QAT is not directly supported in PyTorch
                    # This is a placeholder for when it becomes available
                    print("4-bit quantization-aware training is not currently supported")
                elif q_bits == 16:
                    self.model = torch.quantization.quantize_qat(
                        self.model, {torch.nn.Linear}, dtype=torch.float16
                    )
            else:
                raise ValueError(f"Unsupported quantization type: {q_type}")

        self.model.to(self.device)

# Example usage:
# model = QuantizedHuggingFaceEmbeddingModel("bert-base-uncased", 
#     quantization_config=[("dynamic", 8), ("static", 16)])
# async def get_embedding(text):
#     return await model.embed(text)