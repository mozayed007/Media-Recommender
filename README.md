# Media Recommender

## Project Overview


Media Recommender is an open-source project that uses the MyAnimeList API v2 database to build a recommendation system. The system leverages categorical and numerical features for a machine learning model recommender, content-based filtering, and NLP semantic similarity techniques for synopsis and Media descriptions. This project aims to help users discover new Media that aligns with their tastes.


The techniques used in this project are referenced from various sources including articles, Kaggle, and PapersWithCode.

## Recommendation System

The recommendation system is based on two main components:

1. **Content Filtering**: This model uses categorical and numerical data to recommend similar media based on user preferences.
2. **Semantic Similarity**: A graphRAG system will analyze the description/synopsis feature column (paragraphs) to recommend similar entities. The results from both models are combined to provide a final recommendation list that is semantically similar and aligned with the content-based filtering model.

## Project Progress

- [x] Data collection and processing pipeline using [sample] `anime_scraper.py` (ETL (example) pipeline).
- [x] Designing the recommendation system based on content filtering and semantic similarity.
- [ ] Feature engineering for the recommendation models.
- [ ] Development of the recommendation models.
- [ ] Evaluation of the recommendation models.
- [x] Development of the Recommendation System Query Engine.
- [x] Deployment of the recommendation system.

## Contributing

Contributions are welcome! To contribute, please fork the repository and use a feature branch. Pull requests are also warmly welcome.

## Status

This project is currently under construction. Please check back for updates.
