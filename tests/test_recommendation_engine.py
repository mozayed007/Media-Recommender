import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.embedding_model import HuggingFaceEmbeddingModel
from src.vector_database import MilvusVectorDatabase
from src.recommendation_engine import AnimeRecommendationEngine, OptimizedAnimeRecommendationEngine

def main():
    # Initialize components
    model_name = "dunzhang/stella_en_400M_v5"  # or "dunzhang/stella_en_1.5B_v5"
    embedding_model = HuggingFaceEmbeddingModel(model_name)
    vector_db = MilvusVectorDatabase(embedding_model)
    data_path = "../data/processed/anime_mal_August.csv"  # Update this path to your processed data
    
    # Create and load the recommendation engine
    rec_engine = AnimeRecommendationEngine(embedding_model, vector_db, data_path)
    rec_engine.load_data()

    # Example usage
    query_synopsis = "A thrilling space adventure with giant robots"
    recommendations_by_synopsis = rec_engine.get_recommendations_by_synopsis(query_synopsis, k=5)
    print("Recommendations based on synopsis:")
    for anime_id, title, distance in recommendations_by_synopsis:
        print(f"{anime_id} - {title}: {distance}")

    query_title = "Neon Genesis Evangelion"
    recommendations_by_title = rec_engine.get_recommendations_by_title(query_title, k=5)
    print("\nRecommendations based on title:")
    for anime_id, title, distance in recommendations_by_title:
        print(f"{anime_id} - {title}: {distance}")

if __name__ == "__main__":
    main()

async def main():
    from src.embedding_model import QuantizedHuggingFaceEmbeddingModel
    from src.vector_database import ShardedMilvusVectorDatabase

    model_name = "dunzhang/stella_en_400M_v5"
    embedding_model = QuantizedHuggingFaceEmbeddingModel(model_name)
    vector_db = ShardedMilvusVectorDatabase(embedding_model, num_shards=2)
    data_path = "../data/processed/anime_mal_August.csv"
    processed_data_path = "../data/processed/anime_data_processed.pkl"
    
    rec_engine = OptimizedAnimeRecommendationEngine(embedding_model, vector_db, data_path, processed_data_path)
    await rec_engine.load_data()

    query_synopsis = "A thrilling space adventure with giant robots"
    recommendations_by_synopsis = await rec_engine.get_recommendations_by_synopsis(query_synopsis, k=5)
    print("Recommendations based on synopsis:")
    for anime_id, title, distance in recommendations_by_synopsis:
        print(f"{anime_id} - {title}: {distance}")

    query_title = "Neon Genesis Evangelion"
    recommendations_by_title = await rec_engine.get_recommendations_by_title(query_title, k=5)
    print("\nRecommendations based on title:")
    for anime_id, title, distance in recommendations_by_title:
        print(f"{anime_id} - {title}: {distance}")

    batch_queries = ["A romantic comedy set in high school", "An epic fantasy adventure"]
    batch_results = await rec_engine.batch_recommendations(batch_queries)
    print("\nBatch recommendations:")
    for i, results in enumerate(batch_results):
        print(f"Query {i + 1}:")
        for anime_id, title, distance in results:
            print(f"{anime_id} - {title}: {distance}")

if __name__ == "__main__":
    asyncio.run(main())