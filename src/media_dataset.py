import pandas as pd
import os
from src.abstract_interface_classes import AbstractDataset

class MediaDataset(AbstractDataset):
    """
    A dataset class for managing media data.
    
    Args:
        data_path (str): Path to the dataset file (.csv or .parquet).
        id_col (str): Column name for unique media IDs. Default is 'anime_id'.
        title_col (str): Column name for media titles. Default is 'title'.
        desc_col (str): Column name for media descriptions. Default is 'synopsis'.
    """
    def __init__(self, data_path: str, id_col: str = 'anime_id', title_col: str = 'title', desc_col: str = 'synopsis'):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        file_extension = os.path.splitext(data_path)[1].lower()
        
        if file_extension == '.csv':
            self.data = pd.read_csv(data_path)
        elif file_extension == '.parquet':
            self.data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Please use .csv or .parquet")
            
        # Store column names for flexible access
        self.id_col = id_col
        self.title_col = title_col
        self.desc_col = desc_col
        
        # Validate columns exist
        required_cols = {id_col, title_col, desc_col}
        if not required_cols.issubset(self.data.columns):
            missing_cols = required_cols - set(self.data.columns)
            raise ValueError(f"Missing required columns in {data_path}: {missing_cols}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        # Use the stored column names
        # Ensure ID is returned as the correct type (e.g., int)
        media_id = item[self.id_col]
        try:
            media_id = int(media_id)
        except (ValueError, TypeError):
            # Handle cases where ID might not be purely numeric or is missing
            print(f"Warning: Could not convert ID '{media_id}' to int at index {idx}. Using original value.")
            # Or decide on a default/error handling strategy
            pass 
        return media_id, item[self.title_col], item[self.desc_col]
        
    def get_dataframe(self):
        """Returns the full DataFrame used by this dataset.
        
        Returns:
            pd.DataFrame: The DataFrame containing the media data.
        """
        return self.data