import os
import asyncio
import pickle
from typing import List, Tuple
from torch.utils.data import DataLoader
from abstract_classes import AbstractEmbeddingModel, AbstractVectorDatabase
from anime_dataset import AnimeDataset

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
    

# class OptimizedAnimeRecommendationEngine:
#     def __init__(self, embedding_model: AbstractEmbeddingModel, vector_db: AbstractVectorDatabase, data_path: str, processed_data_path: str, batch_size: int = 64):
#         self.embedding_model = embedding_model
#         self.vector_db = vector_db
#         self.data_path = data_path
#         self.processed_data_path = processed_data_path
#         self.batch_size = batch_size
#         self.anime_data = None
#         self.cache = redis.Redis(host='localhost', port=6379, db=0)

#     def save_processed_data(self):
#         with open(self.processed_data_path, 'wb') as f:
#             pickle.dump(self.anime_data, f)

#     def load_processed_data(self):
#         if os.path.exists(self.processed_data_path):
#             with open(self.processed_data_path, 'rb') as f:
#                 self.anime_data = pickle.load(f)
#             return True
#         return False

#     async def load_data(self):
#         if self.load_processed_data():
#             print("Loaded processed data from file.")
#         else:
#             print("Processing raw data...")
#             dataset = AnimeDataset(self.data_path)
#             dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            
#             ids, titles, synopses = [], [], []
#             for batch in dataloader:
#                 batch_ids, batch_titles, batch_synopses = batch
#                 ids.extend(batch_ids)
#                 titles.extend(batch_titles)
#                 synopses.extend(batch_synopses)
            
#             self.anime_data = list(zip(ids, titles, synopses))
#             await self.vector_db.add_items_batch(ids, titles, synopses)
#             self.save_processed_data()

#     async def get_recommendations_by_synopsis(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
#         cache_key = f"synopsis:{query}:{k}"
#         cached_result = self.cache.get(cache_key)
#         if cached_result:
#             return pickle.loads(cached_result)
        
#         result = await self.vector_db.find_similar_by_synopsis(query, k)
#         self.cache.setex(cache_key, 3600, pickle.dumps(result))  # Cache for 1 hour
#         return result

#     async def get_recommendations_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
#         cache_key = f"title:{title}:{k}"
#         cached_result = self.cache.get(cache_key)
#         if cached_result:
#             return pickle.loads(cached_result)
        
#         result = await self.vector_db.find_similar_by_title(title, k)
#         self.cache.setex(cache_key, 3600, pickle.dumps(result))  # Cache for 1 hour
#         return result

#     async def batch_recommendations(self, queries: List[str], by_synopsis: bool = True, k: int = 5) -> List[List[Tuple[int, str, float]]]:
#         tasks = []
#         for query in queries:
#             if by_synopsis:
#                 task = self.get_recommendations_by_synopsis(query, k)
#             else:
#                 task = self.get_recommendations_by_title(query, k)
#             tasks.append(task)
        
#         return await asyncio.gather(*tasks)