# Media Recommender: Strategic Evolution Plan (2026-2027)

This document outlines the roadmap for transforming the current system into a state-of-the-art, inclusive semantic recommender system, leveraging the latest advancements in AI/ML, RecSys, and modern engineering practices.

## 1. AI/ML Roadmap: Beyond Basic Embeddings

The goal is to move from simple vector similarity to a reasoning-capable recommendation engine.

### Phase 1: Multi-modal Integration (Short-term)

- **Visual-Semantic Alignment**: Implement CLIP-style multi-modal embeddings to index media using both posters/covers and synopses.
- **Cross-Modal Retrieval**: Allow users to search for media using images or "vibe-based" visual queries.
- **Hybrid Scoring**: Combine semantic similarity (NLP) with classic collaborative filtering (User-Item matrices) for better cold-start handling.

### Phase 2: RAG & Explainability (Mid-term)

- **Recommendation Reasoning**: Use Retrieval-Augmented Generation (RAG) to explain *why* a piece of media is recommended (e.g., "Because you liked the themes of redemption in X, you might enjoy the character arc in Y").
- **Dynamic Context Injection**: Use LLMs to rerank vector search results based on real-time user context (e.g., "I'm in the mood for something dark but short").
- **Relational Foundation Models (RFMs)**: Explore pre-trained models on relational structures for zero-shot recommendation across media types.

### Phase 3: Agentic Discovery (Long-term)

- **Agentic Recommendation**: Implement an autonomous agent that can plan complex discovery journeys (e.g., "Find me a manga that has a similar world-building to this anime, then find the live-action drama adaptation if it exists").
- **Conversational Feedback Loops**: Move beyond "thumbs up/down" to conversational refinement ("Less action, more character development").

---

## 2. Data Strategy: The Multi-Media Unified Core

Transitioning from an anime-centric dataset to a unified global media knowledge base.

### Phase 1: Expansion & Ingestion (Short-term)

- **Multi-Source Scrapers**: Implement robust, rate-limited scrapers for:
  - **Manga/Manhawa**: MyAnimeList, Anilist, MangaDex.
  - **Dramas (K/C/J)**: MyDramaList.
  - **Western Media**: TMDB (Movies/TV), Goodreads (Books), RAWG (Games).
- **Unified Schema Enforcement**: Ensure all media follows the `MediaBase` model with strict type safety.

### Phase 2: Semantic ID & Cleaning (Mid-term)

- **Semantic ID Generation (RQ-VAE)**: Replace high-dimensional embeddings with compact Semantic IDs for efficient production-scale ranking and better long-tail item discovery.
- **AI-Driven Data Hygiene**: Use LLMs to normalize genres, tags, and synopses across different sources to prevent "metadata drift".
- **Knowledge Graph Construction**: Build a graph connecting media across types (e.g., *Anime* -> *Adapted From* -> *Manga* -> *Sequel* -> *Movie*).

---

## 3. Backend Evolution: Scale & Resilience

Building a robust infrastructure that can handle millions of items and complex queries.

### Phase 1: High-Performance Vector Ops (Short-term)

- **Vector DB Optimization**: Fine-tune Qdrant/Milvus with HNSW indexing and scalar quantization for sub-100ms latency.
- **Async Processing Pipeline**: Move embedding generation and scraping to background workers (Celery/Redis) to prevent blocking the main API.
- **Streaming Ingestion**: Support real-time indexing of new media as they are scraped.

### Phase 2: User Preference & Feedback (Mid-term)

- **Negative Feedback Support**: Explicitly track and weight "not interested" or "disliked" media in the recommendation algorithm.
- **Distributed Caching**: Implement multi-layer caching (Redis for common queries, local memory for high-frequency metadata).
- **A/B Testing Framework**: Built-in support for testing different recommendation algorithms (e.g., Semantic vs. Hybrid).

### Phase 3: Microservices Architecture (Long-term)

- **Decoupled Services**: Split the backend into `Scraper Service`, `Embedding Service`, `RecSys Engine`, and `API Gateway`.
- **GraphQL Integration**: Use GraphQL to allow the frontend to fetch only the necessary fields for complex multi-media views.

---

## 4. Frontend Experience: Immersive Discovery

Transforming the UI from a list of items into an interactive media portal.

### Phase 1: Multi-Media UI/UX (Short-term)

- **Adaptive Layouts**: Dedicated UI components for different media types (e.g., chapter lists for Manga, episode trackers for TV, platform links for Games).
- **Advanced Filtering**: Unified "Discover" page with cross-media filters (e.g., "Show me everything related to Cyberpunk across Games, Anime, and Movies").
- **Local-First Polish**: Enhance the Watchlist with offline support and instant UI updates.

### Phase 2: Interactive Personalization (Mid-term)

- **Preference Tuner**: A visual, interactive tool to "tune" the recommendation engine (e.g., sliders for Mood, Pacing, Complexity).
- **AI Reasoning Cards**: Display the RAG-generated "Why" behind each recommendation directly on the cards.
- **Multi-Modal Search UI**: Drag-and-drop interface for image-based search.

### Phase 3: The Discovery Journey (Long-term)

- **Cross-Media Hub**: Visualizing the "Media Path" (e.g., "You started with this Anime, which led to this Manga, and now here is the Soundtrack on Spotify").
- **Social Discovery**: Shared watchlists and community-driven recommendation "collections".
- **Agentic Chat Interface**: A natural language search bar that acts as a media concierge.
