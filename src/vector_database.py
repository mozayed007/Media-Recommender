import os
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from llama_index.core import Document, StorageContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.embeddings import BaseEmbedding
from src.abstract_interface_classes import AbstractVectorDatabase, AbstractEmbeddingModel

class EmbeddingModelWrapper(BaseEmbedding):
    def __init__(self, embedding_model: AbstractEmbeddingModel):
        super().__init__()
        self.embedding_model = embedding_model

    def _get_query_embedding(self, query: str) -> List[float]:
        return asyncio.run(self.embedding_model.get_text_embedding(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        return asyncio.run(self.embedding_model.get_text_embedding(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return asyncio.run(self.embedding_model.get_text_embeddings(texts))

class LlamaIndexVectorDatabase(AbstractVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, collection_name: str = "media_items", db_path: str = "llama_index_db"):
        self.embedding_model = EmbeddingModelWrapper(embedding_model)
        self.collection_name = collection_name
        self.db_path = db_path
        self.index = None
        self.storage_context = None

    async def initialize(self):
        os.makedirs(self.db_path, exist_ok=True)
        
        # Create a new storage context if the directory is empty
        if not os.listdir(self.db_path):
            self.storage_context = StorageContext.from_defaults()
        else:
            self.storage_context = StorageContext.from_defaults(persist_dir=self.db_path)
        
        # Create the index with custom embedding model
        self.index = VectorStoreIndex(
            [],
            storage_context=self.storage_context,
            embed_model=self.embedding_model,
        )
        self.storage_context.persist(persist_dir=self.db_path)

    async def add_items(self, ids: List[int], titles: List[str], descriptions: List[str]):
        documents = []
        for id, title, description in zip(ids, titles, descriptions):
            doc = Document(
                text=f"Title: {title}\nDescription: {description}",
                id_=str(id),
                metadata={"title": title, "description": description}
            )
            documents.append(doc)
        
        for doc in documents:
            self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=self.db_path)

    async def find_similar_by_description(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_engine = self.index.as_query_engine(similarity_top_k=k)
        results = query_engine.query(f"Description: {query}")
        
        similar_items = []
        for node in results.source_nodes:
            media_id = int(node.node_id)
            title = node.metadata["title"]
            score = node.score if node.score is not None else 0.0
            similar_items.append((media_id, title, score))
        
        return similar_items

    async def find_similar_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_engine = self.index.as_query_engine(similarity_top_k=k+1)
        results = query_engine.query(f"Title: {title}")
        
        similar_items = []
        for node in results.source_nodes:
            if node.metadata["title"] != title:
                media_id = int(node.node_id)
                node_title = node.metadata["title"]
                score = node.score if node.score is not None else 0.0
                similar_items.append((media_id, node_title, score))
            
            if len(similar_items) == k:
                break
        
        return similar_items

    async def update(self, media_id: int, title: str, description: str):
        doc = Document(
            text=f"Title: {title}\nDescription: {description}",
            id_=str(media_id),
            metadata={"title": title, "description": description}
        )
        self.index.update(doc)
        self.index.storage_context.persist(persist_dir=self.db_path)

    async def get(self, media_id: int) -> Optional[Dict[str, Any]]:
        doc = self.storage_context.docstore.get_document(str(media_id))
        if doc:
            return {"media_id": media_id, "title": doc.metadata["title"], "description": doc.metadata["description"]}
        return None

    async def count(self) -> int:
        return len(self.storage_context.docstore.docs)

    async def clear(self):
        self.storage_context = StorageContext.from_defaults()
        self.index = VectorStoreIndex([], storage_context=self.storage_context, embed_model=self.embedding_model)
        self.index.storage_context.persist(persist_dir=self.db_path)

    def save(self):
        self.index.storage_context.persist(persist_dir=self.db_path)

    async def close(self):
        self.save()

async def create_vector_database(
    embedding_model: AbstractEmbeddingModel,
    collection_name: str = "media_collection",
    db_path: str = "llama_index_db"
) -> LlamaIndexVectorDatabase:
    db = LlamaIndexVectorDatabase(embedding_model, collection_name, db_path)
    await db.initialize()
    return db