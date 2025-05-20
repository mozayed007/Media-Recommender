import os
import sys
import pickle
import asyncio
import pandas as pd  # Added for DataFrame handling
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.abstract_interface_classes import AbstractRecommendationEngine, AbstractEmbeddingModel, AbstractVectorDatabase
from src.media_dataset import MediaDataset
from src.content_filtering import ContentBasedFilter  # Import the new class
from torch.utils.data import Dataset, DataLoader
from lru import LRU  # Correct import for LRU cache
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional  # Added Optional

class OptimizedMediaRecommendationEngine(AbstractRecommendationEngine):
    # custom collate_fn to handle None and non-tensor data types
    def _collate_fn(self, batch):
        ids, titles, descriptions = zip(*batch)
        return (
            [int(i) for i in ids],
            [t if t is not None else "" for t in titles],
            [d if d is not None else "" for d in descriptions]
        )

    def __init__(self, embedding_model: AbstractEmbeddingModel, vector_db: AbstractVectorDatabase, data_path: str, processed_data_path: str, batch_size: int = 64, cache_size: int = 1024, id_col: str = 'MAL_ID', title_col: str = 'Title', desc_col: str = 'Synopsis', content_feature_cols: Optional[Dict[str, list]] = None):  # Use MAL_ID, Title, Synopsis defaults, add content_feature_cols
        super().__init__(embedding_model, vector_db, data_path, batch_size)
        self.processed_data_path = processed_data_path
        self.media_data = None  # This will store list of tuples (id, title, desc)
        self.media_df = None    # This will store the DataFrame
        self.dataset = MediaDataset(data_path, id_col=id_col, title_col=title_col, desc_col=desc_col)
        # Set num_workers=0 for better Windows compatibility
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=self._collate_fn)
        self.description_cache = LRU(cache_size)  # Use LRU from lru package
        self.title_cache = LRU(cache_size)        # Use LRU from lru package
        # Add column names as attributes
        self.id_col = id_col
        self.title_col = title_col
        self.desc_col = desc_col

        # Initialize Content Filter
        self.content_filter = None
        # Use lowercase column names to match the actual data
        self.content_feature_cols = content_feature_cols or {'text': ['genres', 'studios'], 'numeric': ['score']}  # Default features
        # Note: Content filter needs the DataFrame, which is loaded in load_data()

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
        # Load the DataFrame using the dataset loader logic
        print("Loading DataFrame for content filtering and processing...")
        try:
            # Assuming MediaDataset has a method to get the full DataFrame
            # If not, we might need to load it directly using pandas here
            self.media_df = self.dataset.get_dataframe()
            if self.media_df is None or self.media_df.empty:
                raise ValueError("Failed to load DataFrame from MediaDataset.")
            print(f"DataFrame loaded with shape: {self.media_df.shape}")

            # Initialize ContentBasedFilter now that we have the DataFrame
            print("Initializing ContentBasedFilter...")
            try:
                # First attempt with all features
                self.content_filter = ContentBasedFilter(
                    data=self.media_df.copy(),  # Pass a copy to avoid modification issues
                    id_col=self.id_col,  # Pass the correct ID column name
                    text_feature_cols=self.content_feature_cols.get('text', []),
                    numeric_feature_cols=self.content_feature_cols.get('numeric', [])
                )
                print("ContentBasedFilter initialized.")
            except Exception as cf_error:
                print(f"Error initializing ContentBasedFilter with numeric features: {cf_error}")
                print("Retrying with text features only...")
                try:
                    # Fallback: try with only text features
                    self.content_filter = ContentBasedFilter(
                        data=self.media_df.copy(),
                        id_col=self.id_col,
                        text_feature_cols=self.content_feature_cols.get('text', []),
                        numeric_feature_cols=[]
                    )
                    print("ContentBasedFilter initialized with text features only.")
                except Exception as cf_error2:
                    print(f"Error initializing ContentBasedFilter with text features only: {cf_error2}")
                    self.content_filter = None

        except Exception as e:
            print(f"Error loading DataFrame or initializing ContentBasedFilter: {e}")
            # Don't set media_df to None, keep it for filtering even if ContentBasedFilter fails
            self.content_filter = None

        # Try loading processed list data (IDs, titles, descriptions for vector DB)
        if self.load_processed_data():
            print("Loaded processed list data from file.")
        else:
            print("Processing raw data for vector database...")
            ids, titles, descriptions = [], [], []
            # Ensure DataFrame is loaded before iterating dataloader
            if self.media_df is None:
                print("Error: Cannot process data for vector DB as DataFrame failed to load.")
                return  # Stop if DataFrame loading failed

            # Use the DataFrame directly if available, otherwise use dataloader
            # This avoids redundant iteration if DataFrame is already loaded
            try:
                ids = self.media_df[self.id_col].tolist()
                titles = self.media_df[self.title_col].fillna('').tolist()  # Handle potential NaNs
                descriptions = self.media_df[self.desc_col].fillna('').tolist()  # Handle potential NaNs
                self.media_data = list(zip(ids, titles, descriptions))
                self.save_processed_data()  # Save the processed list data
                print("Processed list data saved.")
            except Exception as e:
                print(f"Error processing data from DataFrame for vector DB: {e}")

        # Populate vector database (semantic search part)
        if self.media_data:
            print(f"Adding {len(self.media_data)} items to Milvus collection '{self.vector_db.collection_name}'")
            ids, titles, descriptions = zip(*self.media_data)
            # Ensure IDs are integers for Milvus
            int_ids = [int(i) for i in ids]
            await self.vector_db.add_items(int_ids, list(titles), list(descriptions))
            # Ensure inserted vectors are searchable
            self.vector_db.save()
            # Debug: show count in Milvus
            count = await self.vector_db.count()
            print(f"Milvus collection '{self.vector_db.collection_name}' now contains {count} items")
        else:
            print("No media data to load into Milvus.")

    async def get_recommendations_by_description(self, query: str, k: int = 10) -> List[Tuple[int, str, float]]:
        # Use a more specific cache key with a prefix to avoid collisions
        cache_key = f"desc:{query}:{k}"
        cached_result = self.description_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = await self.vector_db.find_similar_by_description(query, k)
        # Use dictionary-style assignment for LRU cache
        self.description_cache[cache_key] = result
        return result

    async def get_recommendations_by_title(self, title: str, k: int = 10) -> List[Tuple[int, str, float]]:
        # Use a more specific cache key with a prefix to avoid collisions
        cache_key = f"title:{title}:{k}"
        cached_result = self.title_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = await self.vector_db.find_similar_by_title(title, k)
        # Use dictionary-style assignment for LRU cache
        self.title_cache[cache_key] = result
        return result

    def get_content_based_recommendations(self, item_id: int, k: int = 10) -> List[Tuple[int, str, float]]:
        """
        Gets recommendations based on content features (genres, score, etc.).

        Args:
            item_id (int): The ID of the item to get recommendations for.
            k (int): The number of recommendations to return.

        Returns:
            List[Tuple[int, str, float]]: List of (recommended_id, recommended_title, similarity_score).
                                          Returns empty list if content filter not available or item not found.
        """
        if self.content_filter is None:
            print("Error: Content filter is not initialized.")
            return []
        if self.media_df is None:
            print("Error: Media DataFrame not available for retrieving titles.")
            return []

        try:
            recommendations = self.content_filter.get_recommendations(item_id, top_n=k)
            # Enrich with titles
            enriched_recommendations = []
            for rec_id, score in recommendations:
                title = self.get_item_details(rec_id).get(self.title_col, "Title Not Found")
                enriched_recommendations.append((rec_id, title, score))
            return enriched_recommendations
        except Exception as e:
            print(f"Error getting content-based recommendations for item {item_id}: {e}")
            return []

    def get_item_details(self, item_id: int) -> Dict:
        """Retrieves details for a given item ID from the DataFrame."""
        if self.media_df is None or self.id_col not in self.media_df.columns:
            return {}
        try:
            # Ensure DataFrame index is set correctly if needed, or use boolean indexing
            details = self.media_df[self.media_df[self.id_col] == item_id]
            if not details.empty:
                # Convert the first row to a dictionary
                return details.iloc[0].to_dict()
            else:
                return {}  # Item ID not found
        except Exception as e:
            print(f"Error retrieving details for item {item_id}: {e}")
            return {}

    async def batch_recommendations(self, queries: List[str], by_description: bool = True, k: int = 5) -> List[List[Tuple[int, str, float]]]:
        tasks = []
        for query in queries:
            if by_description:
                task = self.get_recommendations_by_description(query, k)
            else:
                task = self.get_recommendations_by_title(query, k)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

    async def get_combined_recommendations(self, item_id: Optional[int] = None, query: Optional[str] = None, title: Optional[str] = None, k: int = 10, alpha: float = 0.5) -> List[Tuple[int, str, float]]:
        """
        Gets combined recommendations from semantic search and content filtering using a weighted approach.
        
        Combines recommendations from two sources:
        1. Semantic search (using title or description query)
        2. Content-based filtering (using item_id to find similar items)
        
        The results are merged with alpha controlling the weight of semantic vs. content results.

        Args:
            item_id (Optional[int]): Item ID for content-based input.
            query (Optional[str]): Description query for semantic search.
            title (Optional[str]): Title for semantic search.
            k (int): Number of recommendations desired.
            alpha (float): Weighting factor for combining scores (0 to 1).
                           alpha=1 -> only semantic, alpha=0 -> only content-based.

        Returns:
            List[Tuple[int, str, float]]: Combined list of (id, title, combined_score).
        """
        print("Warning: Combined recommendation logic is not yet fully implemented.")

        semantic_recs = []
        content_recs = []

        # Get semantic recommendations if query or title provided
        if query:
            semantic_recs = await self.get_recommendations_by_description(query, k * 2)  # Get more to allow for overlap/ranking
        elif title:
            semantic_recs = await self.get_recommendations_by_title(title, k * 2)

        # Get content-based recommendations if item_id provided
        if item_id is not None:
            content_recs = self.get_content_based_recommendations(item_id, k * 2)

        # --- Integration Logic (Simple Example: Weighted Score Combination) ---
        # This is a basic example and needs refinement.
        # Consider normalization, handling duplicates, different ranking strategies.

        combined_scores = {}

        # Add semantic recommendations with weight alpha
        for rec_id, rec_title, score in semantic_recs:
            combined_scores[rec_id] = {'title': rec_title, 'score': score * alpha}

        # Add/update content recommendations with weight (1-alpha)
        for rec_id, rec_title, score in content_recs:
            if rec_id in combined_scores:
                combined_scores[rec_id]['score'] += score * (1 - alpha)
            else:
                combined_scores[rec_id] = {'title': rec_title, 'score': score * (1 - alpha)}

        # Sort by combined score
        sorted_recs = sorted(combined_scores.items(), key=lambda item: item[1]['score'], reverse=True)

        # Format output
        final_recs = [(rec_id, data['title'], data['score']) for rec_id, data in sorted_recs[:k]]

        return final_recs