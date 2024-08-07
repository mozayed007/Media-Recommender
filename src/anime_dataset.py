
import pandas as pd
from abstract_classes import AbstractDataset

class AnimeDataset(AbstractDataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return item['anime_id'], item['title'], item['synopsis']