# Milvus Integration Tests (`test_milvus_integration.py`)

This document describes the integration tests designed to verify the interaction between the recommendation engine and the Milvus vector database.

## Purpose

These tests ensure that:

1. The system can correctly connect to a running Milvus instance (expected to be managed via `docker-compose.yml`).
2. The Milvus collection can be cleared and prepared for testing.
3. Data (including embeddings) can be successfully loaded into the Milvus collection.
4. The recommendation engine can query Milvus to retrieve relevant recommendations based on text descriptions and titles.

## Test Structure

The tests rely on `pytest` and `pytest-asyncio` for asynchronous operations.

### Configuration (`load_config`)

* Reads configuration from `config/main.yaml`.
* Validates that the `vector_database` type is set to `milvus` and that the `data.path` is specified.

### Setup Fixture (`setup_engine`)

This `async` fixture runs once per module (`scope="module"`) and performs the following steps before any tests are executed:

1. **Load Configuration**: Calls `load_config()`.
2. **Initialize Embedding Model**: Creates an instance of `SentenceTransformerEmbeddingModel` based on the config.
3. **Initialize Vector Database**: Creates a `MilvusVectorDatabase` instance, connecting to the Milvus server specified in the config.
4. **Check/Clear Collection**:
    * Prints collection statistics *before* clearing (using `vector_db.client.get_collection_stats`).
    * Calls `vector_db.clear()` to drop and recreate the collection.
    * Prints collection statistics *after* clearing and asserts the collection is empty.
5. **Initialize Recommendation Engine**: Creates an `OptimizedMediaRecommendationEngine` instance, passing the embedding model and vector DB.
6. **Load Data**: Calls `engine.load_data()`, which reads data, generates embeddings, and inserts items into the Milvus collection.
7. **Check Collection After Load**: Prints collection statistics *after* data loading.
8. **Return Engine**: Returns the initialized `engine` instance for use in tests.

*Note: The fixture includes print statements and error handling around `get_collection_stats` to provide visibility into the state of the Milvus collection during setup.*

### Test Functions

* **`test_recommendations_by_description(setup_engine)`**:
  * Takes the initialized `engine` from the fixture.
  * Defines a sample text description.
  * Calls `engine.get_recommendations_by_description()`.
  * Asserts that the results are a list and prints the recommendations (ID, Title, Score).
  * Performs basic type checks on the results.
* **`test_recommendations_by_title(setup_engine)`**:
  * Takes the initialized `engine` from the fixture.
  * Defines a sample title (e.g., "Cowboy Bebop").
  * Calls `engine.get_recommendations_by_title()`.
  * Asserts that the results are a list and prints the recommendations.
  * Asserts that the recommended titles are not the exact same as the query title.
  * Performs basic type checks on the results.

## Running the Tests

Ensure the Milvus Docker containers are running (`docker-compose up -d`). Then, run the tests using `pytest` from the project root:

```bash
pytest tests/test_milvus_integration.py
```

To see verbose output, including the print statements from the fixture:

```bash
pytest -v -s tests/test_milvus_integration.py
```
