import os
import sys
import logging
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from src.embedding_model import OptimizedEmbeddingModel
from src.vector_database import create_vector_database
from src.recommendation_engine import OptimizedMediaRecommendationEngine

# Set up logging directory
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)
# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
logs_dir = os.path.join(current_dir, 'logs')
vector_db_dir = os.path.join(project_root, 'data', 'vector_db')
data_path = os.path.join(project_root, 'data', 'processed', 'anime_mal_Aug24.parquet')
processed_data_dir = os.path.join('data', 'processed_data')

# Create necessary directories
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(vector_db_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)
# Set up logging
logging.basicConfig(
    filename=os.path.join(logs_dir, f'recommendation_engine_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

async def main():
    logging.info('Starting recommendation engine test')

    # Load the embedding model
    #model_name="dunzhang/stella_en_400M_v5"
    model_name = "Alibaba-NLP/gte-large-en-v1.5"
    embedding_model = OptimizedEmbeddingModel(model_name, trust_remote_code=True)
    
    
    # Initialize the vector database
    vector_db = await create_vector_database(
        embedding_model,
        collection_name="anime_items",
        db_path=os.path.join(vector_db_dir, "milvus.db")
    )
    logging.info('Vector database initialized')
    
    # Load anime dataset
    rec_engine = OptimizedMediaRecommendationEngine(
        embedding_model, 
        vector_db, 
        data_path, 
        processed_data_path=os.path.join(processed_data_dir, "anime_data.pkl")
    )
    
    await rec_engine.load_data()
    logging.info('Data loaded')

    # Get recommendations by synopsis
    query_synopsis = "A thrilling space adventure with giant robots"
    recommendations_by_synopsis = await rec_engine.get_recommendations_by_description(query_synopsis, k=5)
    logging.info(f'Recommendations based on synopsis: {recommendations_by_synopsis}')
    print("Recommendations based on synopsis:")
    for anime_id, title, distance in recommendations_by_synopsis:
        print(f"{anime_id} - {title}: {distance}")

    # Get recommendations by title
    query_title = "Neon Genesis Evangelion"
    recommendations_by_title = await rec_engine.get_recommendations_by_title(query_title, k=5)
    logging.info(f'Recommendations based on title: {recommendations_by_title}')
    print("\nRecommendations based on title:")
    for anime_id, title, distance in recommendations_by_title:
        print(f"{anime_id} - {title}: {distance}")

    # Batch recommendations
    batch_queries = ["A romantic comedy set in high school", "An epic fantasy adventure"]
    batch_results = await rec_engine.batch_recommendations(batch_queries)
    logging.info(f'Batch recommendations: {batch_results}')
    print("\nBatch recommendations:")
    for i, results in enumerate(batch_results):
        print(f"Query {i + 1}:")
        for anime_id, title, distance in results:
            print(f"{anime_id} - {title}: {distance}")

    # Save the vector database
    vector_db.save()
    logging.info('Vector database saved')

    # Close the vector database connection
    vector_db.close()
    logging.info('Vector database connection closed')
    logging.info('Recommendation engine test completed')

if __name__ == "__main__":
    asyncio.run(main())