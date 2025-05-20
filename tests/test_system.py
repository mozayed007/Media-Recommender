"""
Test script to verify Media-Recommender functionality after standardization.
This script tests:
1. Data loading with correct column names
2. Content-based filtering with numeric handling
3. Vector database connectivity (if available)
4. End-to-end recommendation generation
"""
import os
import sys
import asyncio
import pandas as pd
from typing import List, Dict, Any

# Add src to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from src.config import (
    ANIME_DATA_FILE,
    PROCESSED_PICKLE_FILE,
    COLUMN_NAMES,
    CONTENT_FEATURES
)

# Import core components
from src.media_dataset import MediaDataset
from src.content_filtering import ContentBasedFilter
from src.embedding_model import MockEmbeddingModel, SentenceTransformerEmbeddingModel
from src.vector_database import MilvusVectorDatabase, create_vector_database
from src.recommendation_engine import OptimizedMediaRecommendationEngine

# Fallback mock vector database if Milvus is not available
class MockVectorDB:
    """Simple mock vector database for testing when Milvus is not available."""
    
    def __init__(self, items=None):
        self.items = items or {}
        print("WARNING: Using mock vector database instead of Milvus")
        
    async def add_items(self, ids, titles, descriptions):
        for id, title, desc in zip(ids, titles, descriptions):
            self.items[id] = {"title": title, "description": desc}
            
    async def find_similar_by_description(self, query, k=10):
        """Return mock results."""
        if not self.items:
            return []
        # Just return first k items as "similar"
        return [(id, item["title"], 0.9) for id, item in list(self.items.items())[:k]]
        
    async def find_similar_by_title(self, title, k=10):
        """Return mock results."""
        if not self.items:
            return []
        # Just return first k items as "similar"
        return [(id, item["title"], 0.8) for id, item in list(self.items.items())[:k]]
        
    def save(self):
        pass
        
    async def count(self):
        return len(self.items)
        
    async def close(self):
        pass

def print_section(title):
    """Print a section title with nice formatting."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

async def test_data_loading():
    """Test loading data from the parquet file."""
    print_section("TESTING DATA LOADING")
    
    try:
        # Check if the data file exists
        if not os.path.exists(ANIME_DATA_FILE):
            print(f"[ERROR] Data file not found at: {ANIME_DATA_FILE}")
            return False
            
        # Try loading with MediaDataset
        print(f"Testing MediaDataset with file: {ANIME_DATA_FILE}")
        dataset = MediaDataset(
            ANIME_DATA_FILE, 
            id_col=COLUMN_NAMES["id"],
            title_col=COLUMN_NAMES["title"],
            desc_col=COLUMN_NAMES["description"]
        )
        
        # Check length
        dataset_length = len(dataset)
        print(f"Dataset loaded successfully with {dataset_length} items")
        
        # Check first item
        first_item = dataset[0]
        print(f"First item ID: {first_item[0]}, Title: {first_item[1][:50]}...")
        
        # Get the DataFrame directly
        df = dataset.get_dataframe()
        print(f"DataFrame shape: {df.shape}")
        
        # Check column types for numeric columns
        numeric_cols = [col for col in CONTENT_FEATURES["numeric"] if col in df.columns]
        print(f"Numeric columns data types:")
        for col in numeric_cols:
            print(f"  - {col}: {df[col].dtype}")
            
        # Check for Int64 columns
        int64_cols = [col for col in df.columns if str(df[col].dtype) == 'Int64']
        if int64_cols:
            print(f"Found Int64 columns: {int64_cols}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Data loading test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_content_filtering():
    """Test content filtering with the real data."""
    print_section("TESTING CONTENT FILTERING")
    
    try:
        # Load DataFrame directly for content filtering
        print(f"Loading data for content filtering from: {ANIME_DATA_FILE}")
        if os.path.exists(ANIME_DATA_FILE):
            if ANIME_DATA_FILE.endswith('.parquet'):
                df = pd.read_parquet(ANIME_DATA_FILE)
            else:
                df = pd.read_csv(ANIME_DATA_FILE)
            
            print(f"Data loaded with {len(df)} rows")
            
            # Initialize content filter with appropriate columns
            text_cols = [col for col in CONTENT_FEATURES["text"] if col in df.columns]
            numeric_cols = [col for col in CONTENT_FEATURES["numeric"] if col in df.columns]
            
            print(f"Using text features: {text_cols}")
            print(f"Using numeric features: {numeric_cols}")
            
            content_filter = ContentBasedFilter(
                data=df, 
                id_col=COLUMN_NAMES["id"],
                text_feature_cols=text_cols,
                numeric_feature_cols=numeric_cols
            )
            
            print(f"Content filter built successfully")
            print(f"Feature matrix shape: {content_filter.feature_matrix.shape}")
            
            # Try getting recommendations for the first anime
            target_id = df[COLUMN_NAMES["id"]].iloc[0]
            print(f"Getting recommendations for anime_id: {target_id}")
            
            recs = content_filter.get_recommendations(target_id, top_n=5)
            
            # Display recommendations
            if recs:
                print("Content-based recommendations:")
                for rec_id, score in recs:
                    rec_title = df.loc[df[COLUMN_NAMES["id"]] == rec_id, COLUMN_NAMES["title"]].iloc[0]
                    print(f"  - ID: {rec_id}, Title: {rec_title[:50]}..., Score: {score:.4f}")
            else:
                print("No recommendations found")
            
            return True
        else:
            print(f"[ERROR] Data file not found")
            return False
    except Exception as e:
        print(f"[ERROR] Content filtering test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_recommendation_engine():
    """Test the full recommendation engine with the real Milvus database."""
    print_section("TESTING RECOMMENDATION ENGINE WITH MILVUS")
    
    try:
        # Initialize a real embedding model and Milvus vector database
        print("Connecting to Milvus at http://localhost:19530...")
        embedding_model = SentenceTransformerEmbeddingModel(model_name="all-MiniLM-L6-v2")
        
        # Use the factory function to create the vector database
        try:
            # First try to use the create_vector_database factory
            vector_db = await create_vector_database(
                "milvus", 
                embedding_model,
                uri="http://localhost:19530",
                collection_name="anime_test_collection",
                dimension=384
            )
            print("Successfully connected to Milvus using the factory function")
        except Exception as e:
            # Fallback to direct instantiation if the factory fails
            print(f"Factory creation failed: {str(e)}")
            print("Falling back to direct MilvusVectorDatabase instantiation...")
            vector_db = MilvusVectorDatabase(
                embedding_model=embedding_model,
                uri="http://localhost:19530", 
                collection_name="anime_test_collection",
                dimension=384
            )
            print("Successfully connected to Milvus using direct instantiation")
        
        # Initialize the recommendation engine
        engine = OptimizedMediaRecommendationEngine(
            embedding_model=embedding_model,
            vector_db=vector_db,
            data_path=ANIME_DATA_FILE,
            processed_data_path=PROCESSED_PICKLE_FILE,
            id_col=COLUMN_NAMES["id"],
            title_col=COLUMN_NAMES["title"],
            desc_col=COLUMN_NAMES["description"],
            content_feature_cols=CONTENT_FEATURES
        )
        
        print("Recommendation engine initialized")
        
        # Load data for testing
        print("Loading data...")
        await engine.load_data()
        
        # Get recommendations by description
        test_query = "A story about friendship and adventure"
        print(f"Testing recommendations by description query: '{test_query}'")
        
        recommendations = await engine.get_recommendations_by_description(test_query, k=5)
        
        if recommendations:
            print("Recommendations returned:")
            for rec_id, title, score in recommendations:
                print(f"  - ID: {rec_id}, Title: {title[:50]}..., Score: {score:.4f}")
        else:
            print("No recommendations returned")
        
        return True
    except Exception as e:
        print(f"[ERROR] Recommendation engine test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def run_tests():
    """Run all tests and report results."""
    print("\nTESTING MEDIA-RECOMMENDER SYSTEM\n")
    print(f"Using data file: {ANIME_DATA_FILE}")
    
    # Store test results
    results = {}
    
    # Test data loading
    results["data_loading"] = await test_data_loading()
    
    # Test content filtering
    results["content_filtering"] = await test_content_filtering()
    
    # Test recommendation engine
    results["recommendation_engine"] = await test_recommendation_engine()
    
    # Print summary
    print_section("TEST RESULTS")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    if all_passed:
        print("\n🎉 All tests passed! The system is working correctly with standardized column names.")
    else:
        print("\n❌ Some tests failed. Review the output above for details.")

if __name__ == "__main__":
    asyncio.run(run_tests())
