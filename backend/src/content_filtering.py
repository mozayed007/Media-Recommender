import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Placeholder for potential future abstract class integration
# from .abstract_interface_classes import RecommenderModel

class ContentBasedFilter:
    """
    Generates recommendations based on content features like genre, studio, score, etc.
    """
    def __init__(self, data: pd.DataFrame, id_col: str = 'anime_id', text_feature_cols: list = ['genres', 'studios'], numeric_feature_cols: list = ['score']):
        """
        Initializes the filter with the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing anime data with features.
            id_col (str): Name of the ID column in the DataFrame. Default is 'anime_id'.
            text_feature_cols (list): List of column names containing text features for TF-IDF.
            numeric_feature_cols (list): List of column names containing numeric features.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
            
        # Check if specified ID column exists, with better error message
        if id_col not in data.columns:
            available_cols = ', '.join(data.columns[:10].tolist())
            raise ValueError(f"DataFrame must contain a '{id_col}' column. Available columns include: {available_cols}...")

        self.id_col = id_col
        self.data = data.set_index(id_col)
        self.text_feature_cols = text_feature_cols
        self.numeric_feature_cols = numeric_feature_cols
        self.feature_matrix = None
        self._build_feature_matrix()

    def _preprocess_text_features(self) -> pd.Series:
        """Combines and preprocesses text features, handling numpy arrays properly."""
        # First, convert any numpy arrays to strings
        processed_df = self.data[self.text_feature_cols].copy()
        
        # Function to convert arrays to strings
        def array_to_str(x):
            import numpy as np
            if isinstance(x, np.ndarray):
                return ' '.join([str(item) for item in x])
            elif isinstance(x, list):
                return ' '.join([str(item) for item in x])
            elif pd.isna(x):
                return ''
            else:
                return str(x)
        
        # Apply conversion to each cell
        for col in self.text_feature_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].apply(array_to_str)
        
        # Now combine the text features into a single string per item
        combined_text = processed_df.fillna('').agg(' '.join, axis=1)
        
        # Simple preprocessing: lowercasing (more can be added)
        combined_text = combined_text.str.lower()
        
        # Replace common separators like commas or pipes if needed, e.g.,
        # combined_text = combined_text.str.replace(',', ' ').str.replace('|', ' ')
        return combined_text

    def _build_feature_matrix(self):
        """Builds the combined feature matrix using TF-IDF for text and scaling for numeric."""
        print("Building feature matrix...")
        # --- Text Features (TF-IDF) ---
        if self.text_feature_cols:
            processed_text = self._preprocess_text_features()
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(processed_text)
            # Convert to DataFrame to easily combine with numeric features
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=self.data.index)
            print(f"TF-IDF matrix shape: {tfidf_df.shape}")
        else:
            tfidf_df = pd.DataFrame(index=self.data.index) # Empty DataFrame if no text features

        # --- Numeric Features (Simple Normalization - more sophisticated scaling can be used) ---
        if self.numeric_feature_cols:
            # First, let's handle each column individually to avoid dtype issues
            numeric_dfs = []
            
            for col in self.numeric_feature_cols:
                if col not in self.data.columns:
                    print(f"Warning: Column '{col}' not found in data. Skipping.")
                    continue
                    
                # Work with a single column at a time
                single_col_df = pd.DataFrame(index=self.data.index)
                
                # Check if the column is the problematic Int64 type
                # If so, convert it to float64 first for processing
                if pd.api.types.is_integer_dtype(self.data[col]) or str(self.data[col].dtype) == 'Int64':
                    # Convert to float64 to handle NULL values
                    col_values = self.data[col].astype('float64')
                else:
                    col_values = self.data[col].copy()
                
                # Handle NaN values
                if col_values.isnull().any():
                    # Calculate mean without NaN values
                    mean_val = col_values.mean()
                    print(f"Warning: Filling NaNs in '{col}' with mean ({mean_val:.2f})")
                    col_values = col_values.fillna(mean_val)
                
                # Normalize to 0-1 range
                min_val = col_values.min()
                max_val = col_values.max()
                if max_val > min_val:
                    col_values = (col_values - min_val) / (max_val - min_val)
                else:
                    # If all values are the same, set to 0.5 (middle of range)
                    col_values = pd.Series(0.5, index=col_values.index)
                
                # Store the processed column
                single_col_df[col] = col_values
                numeric_dfs.append(single_col_df)
            
            # Combine all processed columns
            if numeric_dfs:
                numeric_df = pd.concat(numeric_dfs, axis=1)
                print(f"Numeric features shape: {numeric_df.shape}")
            else:
                numeric_df = pd.DataFrame(index=self.data.index)
                print("Warning: No numeric features were processed.")
        else:
            numeric_df = pd.DataFrame(index=self.data.index) # Empty DataFrame if no numeric features

        # --- Combine Features ---
        # Ensure indices align before concatenating
        self.feature_matrix = pd.concat([tfidf_df, numeric_df], axis=1)
        print(f"Combined feature matrix shape: {self.feature_matrix.shape}")
        # Ensure the matrix only contains numeric types
        self.feature_matrix = self.feature_matrix.astype(float)


    def get_recommendations(self, item_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        """
        Gets recommendations for a given item ID based on feature similarity.

        Args:
            item_id (int): The ID of the item to get recommendations for (uses the ID column specified during initialization).
            top_n (int): The number of recommendations to return.

        Returns:
            list[tuple[int, float]]: A list of tuples, each containing (recommended_item_id, similarity_score).
                                    Returns empty list if item_id not found or error occurs.
        """
        if self.feature_matrix is None or self.feature_matrix.empty:
            print("Error: Feature matrix not built.")
            return []
        if item_id not in self.feature_matrix.index:
            print(f"Error: Item ID {item_id} not found in the dataset index (using '{self.id_col}' as ID column).")
            return []

        try:
            # Get the feature vector for the input item
            item_vector = self.feature_matrix.loc[[item_id]]

            # Calculate cosine similarity between the input item and all other items
            similarities = cosine_similarity(item_vector, self.feature_matrix)[0]

            # Create a Series with similarities, index by item ID
            sim_scores = pd.Series(similarities, index=self.feature_matrix.index)

            # Sort by similarity (descending)
            sim_scores = sim_scores.sort_values(ascending=False)

            # Exclude the item itself and get top N
            # Ensure item_id is treated correctly even if it's not in the sorted list (shouldn't happen)
            if item_id in sim_scores.index:
                sim_scores = sim_scores.drop(item_id)

            top_recommendations = sim_scores.head(top_n)

            # Format as list of tuples (id, score)
            return list(zip(top_recommendations.index, top_recommendations.values))

        except Exception as e:
            print(f"Error calculating recommendations for item {item_id}: {e}")
            return []

# Example Usage (Optional - for testing within the script)
if __name__ == '__main__':
    # Create a dummy DataFrame for testing
    dummy_data = {
        'MAL_ID': [1, 2, 3, 4, 5],
        'Title': ['Anime A', 'Anime B', 'Anime C', 'Anime D', 'Anime E'],
        'Genres': ['Action Adventure', 'Comedy SliceOfLife', 'Action SciFi', 'Comedy Romance', 'Action Drama'],
        'Studios': ['Studio X', 'Studio Y', 'Studio X', 'Studio Z', 'Studio Y'],
        'Score': [8.5, 7.8, 8.9, 7.2, 8.1]
    }
    df = pd.DataFrame(dummy_data)

    print("Initializing ContentBasedFilter...")
    try:
        content_filter = ContentBasedFilter(df, text_feature_cols=['Genres', 'Studios'], numeric_feature_cols=['Score'])
        print("\nFilter Initialized. Feature Matrix Head:")
        print(content_filter.feature_matrix.head())

        item_to_recommend = 1 # Recommend based on 'Anime A'
        print(f"\nGetting recommendations for MAL_ID: {item_to_recommend}")
        recommendations = content_filter.get_recommendations(item_to_recommend, top_n=3)

        print("\nRecommendations:")
        if recommendations:
            for rec_id, score in recommendations:
                rec_title = df.loc[df['MAL_ID'] == rec_id, 'Title'].iloc[0]
                print(f"  - ID: {rec_id}, Title: {rec_title}, Score: {score:.4f}")
        else:
            print("No recommendations found.")

    except Exception as e:
        print(f"\nAn error occurred during example usage: {e}")
