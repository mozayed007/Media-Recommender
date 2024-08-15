import pandas as pd
from src.abstract_interface_classes import AbstractDataset

class MediaDataset(AbstractDataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return item['media_id'], item['title'], item['synopsis']