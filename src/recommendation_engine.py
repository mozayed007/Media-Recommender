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