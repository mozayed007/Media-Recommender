# Media Recommender - System Architecture

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Next.js 16 + React 19 Frontend                    │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │  │
│  │  │  Semantic   │ │   Genre     │ │  Trending   │ │   Watchlist     │  │  │
│  │  │   Search    │ │   Filter    │ │   Section   │ │   (Local)       │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘  │  │
│  │                                                                        │  │
│  │  Tech: TypeScript • Tailwind CSS • Framer Motion • Shadcn UI          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ HTTP/REST API
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     FastAPI Backend (Python)                          │  │
│  │                                                                        │  │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐   │  │
│  │  │  /api/v1/search  │ │  /api/v1/recommend│ │  /api/v1/trending   │   │  │
│  │  │  /api/v1/genres  │ │  /api/v1/details  │ │  /api/v1/discover   │   │  │
│  │  └──────────────────┘ └──────────────────┘ └──────────────────────┘   │  │
│  │                                                                        │  │
│  │  Features: CORS • Async/Await • Pydantic Validation • Lifespan Mgmt   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────────┐
│      RECOMMENDER SERVICE          │ │         EMBEDDING SERVICE             │
│  ┌─────────────────────────────┐  │ │  ┌─────────────────────────────────┐  │
│  │  • Hybrid Recommendation    │  │ │  │  google/embeddinggemma-300m     │  │
│  │  • Content-Based Filtering  │  │ │  │  768-dim Semantic Vectors       │  │
│  │  • Semantic Similarity      │  │ │  │  Sentence-Transformers          │  │
│  │  • Re-ranking Algorithm     │  │ │  │                                 │  │
│  └─────────────────────────────┘  │ │  │  Capabilities:                  │  │
│                                    │ │  │  • Query Embedding              │  │
│  Fallback: Keyword Search          │ │  │  • Document Embedding           │  │
│  Cache: LRU for repeated queries   │ │  │  • Batch Processing             │  │
└───────────────────────────────────┘ └───────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VECTOR DATABASE                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          Qdrant Vector DB                              │  │
│  │                                                                        │  │
│  │  Collection: anime_embeddings                                          │  │
│  │  ├── 24,537 vectors indexed                                           │  │
│  │  ├── 768 dimensions per vector                                        │  │
│  │  ├── HNSW Index (fast ANN search)                                     │  │
│  │  └── Metadata: title, synopsis, genres, score, etc.                   │  │
│  │                                                                        │  │
│  │  Operations: Insert • Search • Filter • Scroll                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│  ┌────────────────────────────┐  ┌────────────────────────────────────────┐ │
│  │    Parquet Data Store      │  │         Data Pipeline                  │ │
│  │                            │  │                                        │ │
│  │  • 29,572 Anime Records    │  │  Raw JSON → Clean → Transform →       │ │
│  │  • Optimized Columnar I/O  │  │  Feature Engineering → Parquet        │ │
│  │  • Fast Pandas Loading     │  │                                        │ │
│  │                            │  │  Custom MAL Scraper (50K+ entries)     │ │
│  └────────────────────────────┘  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

## Data Flow

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  User    │───▶│ Frontend │───▶│  FastAPI │───▶│ Embedding│───▶│ Qdrant   │
│  Query   │    │ (Next.js)│    │  Backend │    │  Model   │    │ Vector DB│
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                      │                              │
                                      │◀─────── Similar Items ───────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │  Re-ranking  │
                               │  + Filtering │
                               └──────────────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │  JSON        │
                               │  Response    │
                               └──────────────┘


## Technology Stack Summary

| Layer      | Technology                                    |
|------------|-----------------------------------------------|
| Frontend   | Next.js 16, React 19, TypeScript, Tailwind    |
| Backend    | FastAPI, Python 3.10+, Uvicorn                |
| ML/AI      | Sentence-Transformers, Google EmbeddingGemma  |
| Vector DB  | Qdrant (HNSW index, 768-dim vectors)          |
| Data       | Pandas, Parquet, Custom ETL Pipeline          |
| DevOps     | Docker (optional), Bun, Conda                 |
```

---

*Diagram created for portfolio showcase - Media Recommender Project*

