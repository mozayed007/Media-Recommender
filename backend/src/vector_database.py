import os
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from pymilvus import MilvusClient, DataType

# Make llama_index import optional so the core functionality works without it
LLAMA_INDEX_AVAILABLE = False
try:
    from llama_index.core import Document, StorageContext
    from llama_index.core.indices import VectorStoreIndex
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.core.storage.index_store import SimpleIndexStore
    from llama_index.core.embeddings import BaseEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    # Create stub classes if llama_index is not available
    class BaseEmbedding:
        pass

from src.abstract_interface_classes import AbstractVectorDatabase, AbstractEmbeddingModel

# Define a Mock class for the embedding model wrapper if needed
class MockEmbeddingModel(AbstractEmbeddingModel):
    """
    A simple mock embedding model for testing purposes that doesn't rely on external libraries.
    """
    def __init__(self, model_name: str = "mock-model", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        print(f"Initialized MockEmbeddingModel with dimension {dimension}")
        
    async def get_text_embedding(self, text: str) -> List[float]:
        """Generate a deterministic but unique mock embedding for a text"""
        # Generate a deterministic embedding based on the text
        import numpy as np
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

# Only define EmbeddingModelWrapper if llama_index is available
if LLAMA_INDEX_AVAILABLE:
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

class MilvusVectorDatabase(AbstractVectorDatabase):
    def __init__(self, embedding_model: AbstractEmbeddingModel, uri: str = "http://localhost:19530", collection_name: str = "media_items", dimension: int = 384):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.dimension = dimension
        self.client = MilvusClient(uri=uri)
        self._initialize_collection()

    def _initialize_collection(self):
        if not self.client.has_collection(self.collection_name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="L2",
                params={"M": 16, "efConstruction": 64}
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
        self.client.load_collection(self.collection_name)

    async def add_items(self, ids: List[int], titles: List[str], descriptions: List[str]):
        if not ids:
            return

        vectors = await self.embedding_model.get_text_embeddings(descriptions)

        data = [
            {"id": mid, "title": title, "description": desc, "vector": vec}
            for mid, title, desc, vec in zip(ids, titles, descriptions, vectors)
        ]

        self.client.insert(collection_name=self.collection_name, data=data)

    async def _search(self, query_vector: List[float], k: int, output_fields: List[str]) -> List[Tuple[int, str, float]]:
        search_params = {
            "metric_type": "L2",
            "params": {"ef": 64},
        }
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=k,
            search_params=search_params,
            output_fields=output_fields
        )

        similar_items = []
        for hit in results[0]:
            media_id = hit['entity']['id']
            title = hit['entity'].get('title', 'N/A')
            score = hit['distance']
            similar_items.append((media_id, title, score))
        
        return similar_items

    async def find_similar_by_description(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_vector = await self.embedding_model.get_text_embedding(query)
        return await self._search(query_vector, k, output_fields=["id", "title"])

    async def find_similar_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        query_vector = await self.embedding_model.get_text_embedding(title)
        results = await self._search(query_vector, k + 1, output_fields=["id", "title"])
        filtered_results = [item for item in results if item[1] != title]
        return filtered_results[:k]

    async def update(self, media_id: int, title: str, description: str):
        vector = await self.embedding_model.get_text_embedding(description)
        data = [{"id": media_id, "title": title, "description": description, "vector": vector}]
        self.client.upsert(collection_name=self.collection_name, data=data)

    async def get(self, media_id: int) -> Optional[Dict[str, Any]]:
        res = self.client.get(
            collection_name=self.collection_name,
            ids=[media_id],
            output_fields=["id", "title", "description"]
        )
        if res:
            item = res[0]
            return {"media_id": item['id'], "title": item['title'], "description": item['description']}
        return None

    async def count(self) -> int:
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        count = int(stats['row_count'])
        return count

    async def clear(self):
        self.client.drop_collection(self.collection_name)
        self._initialize_collection()

    def save(self):
        # Ensure inserted data is flushed and loaded for search
        # Pass collection_name as a string instead of a list to match pymilvus API
        self.client.flush(collection_name=self.collection_name)
        self.client.load_collection(self.collection_name)

    async def close(self):
        self.client.close()

async def create_vector_database(
    db_type: str,
    embedding_model: AbstractEmbeddingModel,
    **kwargs
) -> AbstractVectorDatabase:
    if db_type.lower() == "milvus":
        uri = kwargs.get("uri", "http://localhost:19530")
        collection_name = kwargs.get("collection_name", "media_items")
        dimension = kwargs.get("dimension", 384)
        db = MilvusVectorDatabase(embedding_model, uri=uri, collection_name=collection_name, dimension=dimension)
        return db
    elif db_type.lower() == "llama_index":
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError(
                "The llama_index database type is not available because the llama_index package "
                "is not installed. Please install it with 'pip install llama-index-core llama-index-vector-stores-faiss' "
                "or change the vector_database type to 'milvus' in your config file."
            )
        collection_name = kwargs.get("collection_name", "media_items")
        db_path = kwargs.get("db_path", "llama_index_db")
        db = LlamaIndexVectorDatabase(embedding_model, collection_name, db_path)
        await db.initialize()
        return db
    else:
        raise ValueError(f"Unsupported vector database type: {db_type}")