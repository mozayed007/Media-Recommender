import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBasedFilter:
    """
    Generates recommendations based on content features like genre, studio, score, etc.
    Modernized to handle more complex data types and improved performance.
    """
    def __init__(self, data: pd.DataFrame, id_col: str = 'anime_id', text_feature_cols: list = ['genres', 'studios'], numeric_feature_cols: list = ['score']):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
            
        self.id_col = id_col
        # Ensure ID column exists
        if id_col not in data.columns:
            # Try 'id' if 'anime_id' fails
            if 'id' in data.columns:
                self.id_col = 'id'
                id_col = 'id'
            else:
                raise ValueError(f"DataFrame must contain an ID column. Found: {data.columns.tolist()}")

        self.data = data.set_index(id_col)
        self.text_feature_cols = [c for c in text_feature_cols if c in self.data.columns]
        self.numeric_feature_cols = [c for c in numeric_feature_cols if c in self.data.columns]
        self.feature_matrix = None
        self._build_feature_matrix()

    def _preprocess_text_features(self) -> pd.Series:
        processed_df = self.data[self.text_feature_cols].copy()
        
        def array_to_str(x):
            if isinstance(x, (np.ndarray, list)):
                return ' '.join([str(item) for item in x])
            elif pd.isna(x):
                return ''
            else:
                return str(x)
        
        for col in self.text_feature_cols:
            processed_df[col] = processed_df[col].apply(array_to_str)
        
        combined_text = processed_df.fillna('').agg(' '.join, axis=1)
        return combined_text.str.lower()

    def _build_feature_matrix(self):
        print("Building feature matrix for content filtering...")
        
        parts = []
        
        # Text Features
        if self.text_feature_cols:
            processed_text = self._preprocess_text_features()
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = tfidf.fit_transform(processed_text)
            parts.append(tfidf_matrix.toarray())

        # Numeric Features
        if self.numeric_feature_cols:
            numeric_data = self.data[self.numeric_feature_cols].fillna(0).values.astype(float)
            # Make a copy to ensure the array is writable
            numeric_data = numeric_data.copy()
            
            # Simple min-max scaling
            for i in range(numeric_data.shape[1]):
                col = numeric_data[:, i]
                min_val, max_val = col.min(), col.max()
                if max_val > min_val:
                    numeric_data[:, i] = (col - min_val) / (max_val - min_val)
            parts.append(numeric_data)

        if parts:
            self.feature_matrix = np.hstack(parts)
        else:
            raise ValueError("No features available to build matrix.")

    def recommend(self, item_id: int, top_n: int = 10) -> pd.DataFrame:
        if item_id not in self.data.index:
            return pd.DataFrame()

        idx = self.data.index.get_loc(item_id)
        target_features = self.feature_matrix[idx].reshape(1, -1)
        
        similarities = cosine_similarity(target_features, self.feature_matrix).flatten()
        
        # Get top N indices excluding the item itself
        related_indices = similarities.argsort()[-(top_n+1):-1][::-1]
        
        results = self.data.iloc[related_indices].copy()
        results['similarity'] = similarities[related_indices]
        results = results.reset_index()
        
        return results
