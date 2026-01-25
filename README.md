# Media Recommender 🚀

An inclusive, high-performance semantic media recommendation system. Unified AI-powered discovery for **Anime, Manga, Manhawa, Dramas (K/C/J), Novels, TV Series, Movies, Music, and Games**.

Built with a hybrid approach combining **Modern RAG (Retrieval-Augmented Generation)** and **Classical Content-Based Filtering** for the ultimate discovery experience.

## ✨ Features

- 🔍 **Inclusive Semantic Search**: Find any media using natural language (e.g., "emotional high school romance with a sad ending" or "epic dark fantasy with ninjas").
- 🤖 **Hybrid Recommendation Engine**: Combines vector similarity (semantic) with content-based filtering for high-precision discovery.
- ⚡ **Multi-Media Support**: Unified backend architecture designed for cross-media recommendations.
- 🎨 **Modern UI/UX**: Responsive discovery interface built with Next.js 15, Tailwind CSS, and Shadcn UI.
- 🧬 **Vector Engine**: Powered by **Qdrant** and **Gemma** embeddings for deep semantic understanding of plots and themes.
- 📊 **Trending & Discovery**: Real-time trending sections and AI-curated "Discover" categories.
- 🔖 **Watchlist**: Local-first watchlist management to save your next favorites.

## 🏗️ Architecture

### Backend (FastAPI & Python)

- **Framework**: FastAPI for high-performance asynchronous API endpoints.
- **Vector DB**: Qdrant (local/Docker) for high-speed semantic retrieval.
- **Embeddings**: `google/embeddinggemma-300m` via Sentence-Transformers.
- **Data Engine**: Pandas & Parquet for optimized local data processing and fast loading.
- **Hybrid Logic**: Custom re-ranking based on scores, popularity, and semantic relevance.

### Frontend (Next.js & Bun)

- **Framework**: Next.js 15 (App Router) with TypeScript.
- **Runtime**: Bun for ultra-fast development and package management.
- **Styling**: Tailwind CSS + Shadcn UI for a sleek, dark-mode focused aesthetic.
- **Animations**: Framer Motion for smooth transitions and interactive elements.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Bun](https://bun.sh/) (for frontend)
- [Docker](https://www.docker.com/) (optional, for Qdrant server mode)

### Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/Media-Recommender.git
   cd Media-Recommender
   ```

2. **Backend Setup**

   ```bash
   cd backend
   pip install -r requirements.txt
   # Copy .env.example to .env and configure your keys
   python main.py
   ```

3. **Frontend Setup**

   ```bash
   cd ../frontend
   bun install
   bun dev
   ```

## 🗺️ Project Status

- [x] **Phase 1: Anime Foundation**: Core semantic engine and UI built on anime data.
- [x] **Phase 2: Discovery & UX**: Trending, Genres, Discover categories, and Watchlist.
- [x] **Phase 3: Resilient Architecture**: Decoupled service initialization and keyword fallback.
- [x] **Phase 4: Generic Media Migration**: Backend refactored to support all media types.
- [ ] **Next**: Data ingestion pipelines for Manga, Movies, and Dramas.
- [ ] **Next**: Cross-media semantic mapping (e.g., "Find movies like this anime").
- [ ] **Next**: User accounts & cloud-synced watchlist.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: This project is developed for **studying and academic purposes only**. The data used is for educational demonstration of recommendation system techniques and semantic search implementations.
