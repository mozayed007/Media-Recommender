# tests/test_milvus_integration.py
import asyncio
import yaml
import pytest
import pytest_asyncio
import os
import sys
import random
import pandas as pd
import pickle # Add pickle import

# Attempt to import nltk and download necessary data
try:
    import nltk
    nltk.download('punkt', quiet=True) # Download sentence tokenizer data
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not found. Description summarization for testing will be skipped.")

# Add src directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the MockEmbeddingModel as a fallback if SentenceTransformerEmbeddingModel isn't available
try:
    from src.embedding_model import SentenceTransformerEmbeddingModel
except ImportError:
    from src.embedding_model import MockEmbeddingModel as SentenceTransformerEmbeddingModel
    print("Using MockEmbeddingModel because SentenceTransformerEmbeddingModel could not be imported")

from src.vector_database import create_vector_database, MilvusVectorDatabase
from src.recommendation_engine import OptimizedMediaRecommendationEngine

# --- Configuration Loading ---
CONFIG_PATH = 'config/main.yaml'

def load_config():
    """Loads configuration from the YAML file."""
    if not os.path.exists(CONFIG_PATH):
        pytest.fail(f"Configuration file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    # Basic validation
    if 'vector_database' not in config or config['vector_database'].get('type') != 'milvus':
        pytest.fail(f"Test requires vector_database type 'milvus' in {CONFIG_PATH}")
    if 'data' not in config or 'path' not in config['data']:
        pytest.fail(f"Test requires data.path to be set in {CONFIG_PATH}")
    return config

# --- Fixtures ---
@pytest_asyncio.fixture(scope="module")
async def setup_engine():
    """Fixture to set up the recommendation engine, load data, and prepare test data."""
    print("\n--- Starting Test Module Setup ---")
    config = load_config()
    data_cfg = config['data']
    emb_cfg = config['embedding_model']
    vec_db_cfg = config['vector_database']
    engine_cfg = config['recommendation_engine']

    print(f"[Setup] Using config file: {CONFIG_PATH}")
    print(f"[Setup] Data Path: {data_cfg['path']}")
    print(f"[Setup] Embedding Model: {emb_cfg['type']} ({emb_cfg['model_name']})")
    print(f"[Setup] Vector DB: {vec_db_cfg['type']} (Collection: {vec_db_cfg.get('collection_name', 'test_anime_items')})")

    print("\n[Setup] Initializing embedding model...")
    if emb_cfg['type'] == 'sentence_transformer':
        embedding_model = SentenceTransformerEmbeddingModel(model_name=emb_cfg['model_name'])
        print(f"[Setup] Embedding model device: {embedding_model.device}")
    else:
        pytest.fail(f"Unsupported embedding model type: {emb_cfg['type']}")

    print("[Setup] Initializing vector database (Milvus)...")
    vector_db = await create_vector_database(
        db_type=vec_db_cfg['type'],
        embedding_model=embedding_model,
        uri=vec_db_cfg.get('uri'),
        collection_name=vec_db_cfg.get('collection_name', 'test_anime_items'),
        dimension=vec_db_cfg.get('dimension')
    )
    assert isinstance(vector_db, MilvusVectorDatabase), "Vector DB is not MilvusVectorDatabase"
    print(f"[Setup] MilvusVectorDatabase initialized for collection '{vector_db.collection_name}'")

    # --- Inspection Point 1: Before Clearing ---
    print("\n[Setup] Checking Milvus collection status before clearing...")
    try:
        if vector_db.client.has_collection(vector_db.collection_name):
            stats_before = vector_db.client.get_collection_stats(vector_db.collection_name)
            print(f"[Setup] Milvus collection '{vector_db.collection_name}' stats BEFORE clear: {stats_before}")
        else:
            print(f"[Setup] Milvus collection '{vector_db.collection_name}' does not exist before clear.")
    except Exception as e:
        print(f"[Setup] Could not get stats before clear (might be first run or connection issue): {e}")

    print(f"[Setup] Clearing Milvus collection '{vector_db.collection_name}'...")
    await vector_db.clear()  # This drops and recreates the collection
    print(f"[Setup] Milvus collection '{vector_db.collection_name}' cleared.")

    # --- Inspection Point 2: After Clearing ---
    print("\n[Setup] Checking Milvus collection status after clearing...")
    try:
        if vector_db.client.has_collection(vector_db.collection_name):
            stats_after_clear = vector_db.client.get_collection_stats(vector_db.collection_name)
            print(f"[Setup] Milvus collection '{vector_db.collection_name}' stats AFTER clear: {stats_after_clear}")
            row_count = stats_after_clear.get('row_count', 0) if isinstance(stats_after_clear, dict) else 0
            assert row_count == 0, f"Collection should be empty after clear, but got row_count: {row_count}"
            print("[Setup] Confirmed collection is empty after clear.")
        else:
            print(f"[Setup] Milvus collection '{vector_db.collection_name}' does not exist after clear (unexpected).")
            pytest.fail("Collection should exist after clear operation.")
    except Exception as e:
        print(f"[Setup] Could not get stats after clear: {e}")
        pytest.fail(f"Failed to get stats after clear: {e}")

    print("\n[Setup] Initializing recommendation engine...")
    engine = OptimizedMediaRecommendationEngine(
        embedding_model=embedding_model,
        vector_db=vector_db,
        data_path=data_cfg['path'],
        processed_data_path=data_cfg['processed_path'],
        batch_size=engine_cfg['batch_size'],
        cache_size=engine_cfg['cache_size'],
        id_col=data_cfg['id_col'],
        title_col=data_cfg['title_col'],
        desc_col=data_cfg['desc_col']
    )
    print("[Setup] Recommendation engine initialized.")

    print("\n[Setup] Loading data into Milvus (this may take time)...")
    await engine.load_data()
    print("[Setup] Data loading process complete.")

    # --- Inspection Point 3: After Loading ---
    print("\n[Setup] Checking Milvus collection status after loading...")
    final_row_count = 0
    try:
        if vector_db.client.has_collection(vector_db.collection_name):
            stats_after_load = vector_db.client.get_collection_stats(vector_db.collection_name)
            print(f"[Setup] Milvus collection '{vector_db.collection_name}' stats AFTER load: {stats_after_load}")
            final_row_count = stats_after_load.get('row_count', 0) if isinstance(stats_after_load, dict) else 0
            assert final_row_count > 0, "Data should have been loaded"
            print(f"[Setup] Confirmed collection contains {final_row_count} items after load.")
        else:
            print(f"[Setup] Milvus collection '{vector_db.collection_name}' does not exist after load (unexpected).")
            pytest.fail("Collection should exist after load operation.")
    except Exception as e:
        print(f"[Setup] Could not get stats after load: {e}")
        pytest.fail(f"Failed to get stats after load: {e}")

    # --- Load data for randomized testing directly from Milvus ---
    print("\n[Setup] Fetching sample data from Milvus for randomized test queries...")
    test_data = []
    try:
        if final_row_count > 0:
            # Fetch a sample of items directly from Milvus
            # Use the actual field names defined in the Milvus schema
            output_fields = ["id", "title", "description"] # Corrected field names
            # Fetching a limited number (e.g., 1000) to avoid pulling the entire dataset
            sample_limit = min(1000, final_row_count) # Fetch up to 1000 items or total count
            print(f"[Setup] Querying Milvus for up to {sample_limit} items...")
            # Use the hardcoded primary key field name "id" as defined in MilvusVectorDatabase
            pk_field = "id"
            query_expr = f"{pk_field} >= 0" # Simple expression to get all entities

            milvus_sample_data = vector_db.client.query(
                collection_name=vector_db.collection_name,
                filter=query_expr, # Filter expression
                output_fields=output_fields,
                limit=sample_limit
            )

            if milvus_sample_data:
                # Convert the list of dictionaries from Milvus into the expected format
                # Use the correct field name "description" when accessing Milvus results
                # Map it back to engine.desc_col ('synopsis') for consistency within the test data structure
                test_data = [
                        {engine.title_col: item["title"], engine.desc_col: item["description"]}
                        for item in milvus_sample_data
                        if item.get("title") is not None and item.get("description") is not None
                    ]
                print(f"[Setup] Loaded {len(test_data)} valid items from Milvus for randomized testing.")
            else:
                print("[Setup] Warning: Milvus query returned no sample data.")

        if not test_data:
                print("[Setup] Warning: No valid test data loaded from Milvus. Randomized tests might fail or be skipped.")

    except Exception as e:
        print(f"[Setup] Error fetching sample data from Milvus: {e}. Randomized tests might fail or be skipped.")
        # Don't fail the setup, allow tests to run with fixed queries if needed

    print("--- Test Module Setup Complete ---")
    # Return both engine and test data
    return {"engine": engine, "test_data": test_data}

# --- Test Functions ---
@pytest.mark.asyncio
async def test_recommendations_by_description(setup_engine):
    """Tests getting recommendations by description using a randomized query."""
    engine_data = setup_engine
    engine = engine_data["engine"]
    test_data = engine_data["test_data"]

    print("\n--- Running Test: test_recommendations_by_description ---")

    if not test_data:
        pytest.skip("Skipping randomized description test due to lack of test data.")

    # Select a random item
    random_item = random.choice(test_data)
    original_desc = random_item[engine.desc_col]

    # Use NLTK to get the first sentence as the query, fallback to full description
    test_query_desc = original_desc
    if NLTK_AVAILABLE:
        try:
            sentences = nltk.sent_tokenize(original_desc)
            if sentences:
                test_query_desc = sentences[0]
                print(f"[Test] Using first sentence as query: '{test_query_desc}'")
            else:
                print("[Test] Could not extract sentence, using full description.")
        except Exception as e:
            print(f"[Test] NLTK error tokenizing description: {e}. Using full description.")
    else:
        print("[Test] NLTK not available, using full description for query.")

    print(f"[Test] Original Description: '{original_desc[:200]}...'") # Print truncated original
    print(f"[Test] Querying recommendations for description: '{test_query_desc}'")
    recommendations = await engine.get_recommendations_by_description(test_query_desc, k=5)

    assert recommendations is not None, "Recommendations should not be None"
    assert isinstance(recommendations, list), "Recommendations should be a list"

    print("[Test] Results (Description):")
    if recommendations:
        assert len(recommendations) <= 5, "Should return at most k recommendations"
        for rec_id, title, score in recommendations:
            print(f"  ID: {rec_id}, Title: {title}, Score: {score:.4f}")
            assert isinstance(rec_id, int), "ID should be an integer"
            assert isinstance(title, str), "Title should be a string"
            assert isinstance(score, float), "Score should be a float"
    else:
        print("  No recommendations found for this description query.")
    print("--- Test Complete: test_recommendations_by_description ---")

@pytest.mark.asyncio
async def test_recommendations_by_title(setup_engine):
    """Tests getting recommendations by title using a randomized query."""
    engine_data = setup_engine
    engine = engine_data["engine"]
    test_data = engine_data["test_data"]

    print("\n--- Running Test: test_recommendations_by_title ---")

    if not test_data:
        pytest.skip("Skipping randomized title test due to lack of test data.")

    # Select a random item
    random_item = random.choice(test_data)
    test_query_title = random_item[engine.title_col]

    print(f"[Test] Querying recommendations for title: '{test_query_title}'")
    recommendations = await engine.get_recommendations_by_title(test_query_title, k=5)

    assert recommendations is not None, "Recommendations should not be None"
    assert isinstance(recommendations, list), "Recommendations should be a list"

    print("[Test] Results (Title):")
    if recommendations:
        assert len(recommendations) <= 5, "Should return at most k recommendations"
        found_other_title = False
        for rec_id, title, score in recommendations:
            print(f"  ID: {rec_id}, Title: {title}, Score: {score:.4f}")
            assert isinstance(rec_id, int), "ID should be an integer"
            assert isinstance(title, str), "Title should be a string"
            assert isinstance(score, float), "Score should be a float"
            # It's possible Milvus returns the exact item if it's the closest match, 
            # especially if the dataset has duplicates or very similar titles.
            # We'll just check if *at least one* different title is found if possible.
            if title != test_query_title:
                found_other_title = True
        # Assert that if recommendations were found, at least one is different (unless only 1 result and it's the same)
        if len(recommendations) > 1:
                assert found_other_title, f"Should have found at least one different title for query '{test_query_title}' if multiple results returned."
        elif len(recommendations) == 1 and recommendations[0][1] == test_query_title:
                print(f"[Test] Only one recommendation found, and it matches the query title '{test_query_title}'.")
        elif len(recommendations) == 1 and recommendations[0][1] != test_query_title:
                found_other_title = True # Correctly found a different title

    else:
        print(f"  No recommendations found for title query '{test_query_title}'.")
    print("--- Test Complete: test_recommendations_by_title ---")

# Add more tests as needed (e.g., edge cases, different k values)
