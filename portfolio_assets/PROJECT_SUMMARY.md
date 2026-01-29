# Media Recommender - Portfolio Project Summary

## 🎯 Project Overview

**AI-Powered Media Recommendation Engine** that uses semantic understanding to help users discover anime (with movies, manga, and more coming soon). Built with modern ML/AI techniques and a beautiful, responsive UI.

---

## 🔑 Key Highlights

### Technical Achievements
- **Vector Search at Scale**: 24,537+ anime indexed with 768-dimensional semantic embeddings
- **Sub-second Queries**: HNSW index enables real-time similarity search
- **Hybrid Recommendations**: Combines vector similarity with content-based filtering
- **Production-Ready API**: FastAPI with health checks, error handling, and CORS

### User Experience
- **Natural Language Search**: "Find me a sad romance anime with a strong female lead"
- **Smart Recommendations**: Click any title to find similar anime instantly
- **Curated Discovery**: AI-powered category recommendations
- **Personal Watchlist**: Save favorites with local storage persistence

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS, Framer Motion, Shadcn UI |
| **Backend** | Python 3.10+, FastAPI, Uvicorn, Pydantic |
| **ML/AI** | Sentence-Transformers, Google EmbeddingGemma-300M (768-dim) |
| **Database** | Qdrant Vector DB (HNSW index), Parquet data storage |
| **Data Pipeline** | Pandas, Custom MAL web scraper, ETL automation |
| **DevOps** | Docker, Bun, Conda, Git |

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Anime Records | 29,572 |
| Indexed Vectors | 24,537 |
| Embedding Dimensions | 768 |
| API Response Time | <500ms |
| Frontend Load Time | <3s |

---

## 🎨 Features Showcase

### 1. Semantic Search
Users can search using natural language descriptions instead of exact titles:
- "A dark psychological thriller with mind games"
- "Wholesome slice of life with cute characters"
- "Epic fantasy adventure with magic and dragons"

### 2. Smart Recommendations
Click "Find Similar" on any anime to get AI-powered recommendations based on:
- Plot and theme similarity (via embeddings)
- Genre overlap
- User ratings and popularity

### 3. Discover Mode
AI-curated categories like:
- "Action Masterpieces"
- "Emotional Journeys"
- "Hidden Gems"

### 4. Modern UI/UX
- Dark theme with purple/pink accent colors
- Smooth animations with Framer Motion
- Responsive grid layout (2-6 columns)
- Modal details with synopsis and metadata

---

## 🏗️ Architecture Highlights

1. **Decoupled Services**: Frontend and Backend can scale independently
2. **Async Everything**: Non-blocking I/O throughout the stack
3. **Graceful Degradation**: Keyword fallback when vector search unavailable
4. **Configurable**: Environment-based settings for different deployments

---

## 📁 Project Structure

```
Media-Recommender/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI endpoints
│   │   ├── core/         # Config, DB clients, embedding
│   │   ├── models/       # Pydantic schemas
│   │   └── services/     # Business logic
│   ├── data/             # Parquet data & Qdrant storage
│   └── main.py           # Application entry point
├── frontend/
│   ├── src/
│   │   ├── app/          # Next.js pages
│   │   ├── components/   # React components
│   │   └── types/        # TypeScript definitions
│   └── package.json
└── docs/                  # Documentation
```

---

## 🚀 Future Roadmap

- [ ] Multi-media support (Movies, Manga, Dramas)
- [ ] Cross-media recommendations ("Find movies like this anime")
- [ ] User accounts with cloud-synced watchlists
- [ ] Advanced filters (year, rating, studio)
- [ ] Mobile app (React Native)

---

## 📄 Links

- **GitHub**: [Repository Link]
- **Live Demo**: [Deployment URL]
- **API Docs**: http://localhost:8000/docs

---

*Built with ❤️ for studying recommendation systems and semantic search*

