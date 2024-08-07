import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.embedding_model import HuggingFaceEmbeddingModel
from src.vector_database import MilvusVectorDatabase
from src.recommendation_engine import AnimeRecommendationEngine

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