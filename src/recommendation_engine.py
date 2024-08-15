import os
import sys
import pickle
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.abstract_interface_classes import AbstractRecommendationEngine, AbstractEmbeddingModel, AbstractVectorDatabase
from src.media_dataset import MediaDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List, Tuple, Dict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class OptimizedMediaRecommendationEngine(AbstractRecommendationEngine):
    def __init__(self, embedding_model: AbstractEmbeddingModel, vector_db: AbstractVectorDatabase, data_path: str, processed_data_path: str, batch_size: int = 64, cache_size: int = 1024):
        super().__init__(embedding_model, vector_db, data_path, batch_size)
        self.processed_data_path = processed_data_path
        self.media_data = None
        self.dataset = MediaDataset(data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.description_cache = LRUCache(cache_size)
        self.title_cache = LRUCache(cache_size)

    def save_processed_data(self):
        with open(self.processed_data_path, 'wb') as f:
            pickle.dump(self.media_data, f)

    def load_processed_data(self):
        if os.path.exists(self.processed_data_path):
            with open(self.processed_data_path, 'rb') as f:
                self.media_data = pickle.load(f)
            return True
        return False

    async def load_data(self):
        if self.load_processed_data():
            print("Loaded processed data from file.")
        else:
            print("Processing raw data...")
            ids, titles, descriptions = [], [], []
            for batch in self.dataloader:
                batch_ids, batch_titles, batch_descriptions = batch
                ids.extend(batch_ids)
                titles.extend(batch_titles)
                descriptions.extend(batch_descriptions)
            
            self.media_data = list(zip(ids, titles, descriptions))
            await self.vector_db.add_items_batch(ids, titles, descriptions)
            self.save_processed_data()

    async def get_recommendations_by_description(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        cache_key = f"{query}:{k}"
        cached_result = self.description_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = await self.vector_db.find_similar_by_description(query, k)
        self.description_cache.put(cache_key, result)
        return result

    async def get_recommendations_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        cache_key = f"{title}:{k}"
        cached_result = self.title_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = await self.vector_db.find_similar_by_title(title, k)
        self.title_cache.put(cache_key, result)
        return result

    async def batch_recommendations(self, queries: List[str], by_description: bool = True, k: int = 5) -> List[List[Tuple[int, str, float]]]:
        tasks = []
        for query in queries:
            if by_description:
                task = self.get_recommendations_by_description(query, k)
            else:
                task = self.get_recommendations_by_title(query, k)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

# The AsyncOptimizedMediaRecommendationEngine class is no longer needed as we've made OptimizedMediaRecommendationEngine async