import pandas as pd
import numpy as np
import os
import asyncio
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from app.core.milvus_db import MilvusVectorDB
from app.core.qdrant_db import QdrantVectorDB
from app.core.vector_db import VectorDBInterface
from app.models.media import MediaRecommendation, RecommendationResponse, RecommendationRequest
from app.core.embedding import GemmaEmbeddingModel, MockEmbeddingModel
from app.core.classical_ml import ContentBasedFilter
from app.core.agents import interpret_query
from app.core.logging_config import setup_logging

logger = setup_logging("recommender")

class RecommenderService:
    def __init__(self, provider: str = "qdrant"):
        self.embedding_model = None
        self.vector_db: Optional[VectorDBInterface] = None
        self.content_filter = None
        self.df = None
        self.is_initialized = False
        self.provider = provider
        self._lock = asyncio.Lock()

    async def initialize(self):
        async with self._lock:
            if self.is_initialized:
                return

            logger.info(f"Initializing Recommender Service with {self.provider}...")
            
            try:
                # 1. Load data (CRITICAL)
                await self._load_data()
                
                # 2. Initialize Content-based Filter (DEPENDS ON DATA)
                if self.df is not None and not self.df.empty:
                    try:
                        self.content_filter = ContentBasedFilter(self.df, id_col='media_id')
                        logger.info("Content-based filter initialized.")
                    except Exception as e:
                        logger.error(f"Failed to initialize content filter: {e}")

                # 3. Initialize Vector DB (NON-CRITICAL for basic functionality)
                try:
                    await self._init_vector_db()
                    logger.info("Vector DB initialized.")
                except Exception as e:
                    logger.error(f"Failed to initialize Vector DB (Search features will be limited): {e}")
                    self.vector_db = None
                
                # 4. Initialize Embedding Model (NON-CRITICAL for basic functionality)
                try:
                    await self._init_embedding_model()
                    logger.info("Embedding model initialized.")
                except Exception as e:
                    logger.error(f"Failed to initialize embedding model: {e}")
                    self.embedding_model = MockEmbeddingModel(dimension=settings.EMBEDDING_DIMENSION)

                self.is_initialized = True
                logger.info("Recommender Service partially/fully initialized.")
                
            except Exception as e:
                logger.error(f"Critical failure during Recommender Service initialization: {e}", exc_info=True)
                raise

    async def _load_data(self):
        """Load the media dataset from parquet or json."""
        try:
            data_path = settings.absolute_data_path
            if not os.path.exists(data_path):
                logger.error(f"Data path does not exist: {data_path}")
                self.df = pd.DataFrame(columns=['media_id', 'title', 'synopsis', 'genres', 'score', 'media_type'])
            elif data_path.endswith('.parquet'):
                self.df = pd.read_parquet(data_path)
            else:
                self.df = pd.read_json(data_path)
            
            # Rename columns for consistency
            column_mapping = {
                'mean': 'score',
                'media_type': 'media_type', # Already consistent or will be
                'num_episodes': 'episodes',
                'main_picture_large': 'main_picture',
                'anime_id': 'media_id',
                'id': 'media_id'
            }
            # Only rename if column exists
            rename_map = {k: v for k, v in column_mapping.items() if k in self.df.columns}
            if rename_map:
                self.df.rename(columns=rename_map, inplace=True)
                
            # Default media_type to anime if missing (legacy)
            if 'media_type' not in self.df.columns:
                self.df['media_type'] = 'anime'
            
            # Fallback for main_picture if large is missing but medium is present
            if 'main_picture' not in self.df.columns or self.df['main_picture'].isna().all():
                if 'main_picture_medium' in self.df.columns:
                    self.df['main_picture'] = self.df['main_picture_medium']
            
            # Set index to media_id for O(1) lookup
            if 'media_id' in self.df.columns:
                self.df.set_index('media_id', inplace=True, drop=False)
                # Ensure unique index
                if not self.df.index.is_unique:
                    self.df = self.df[~self.df.index.duplicated(keep='first')]
                
            logger.info(f"Loaded {len(self.df)} media records. Columns: {self.df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.df = pd.DataFrame(columns=['media_id', 'title', 'synopsis', 'genres', 'score', 'media_type'])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _init_vector_db(self):
        """Initialize the vector database with retry logic."""
        if self.provider == "qdrant":
            self.vector_db = QdrantVectorDB(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                path=settings.qdrant_storage_path
            )
        else:
            self.vector_db = MilvusVectorDB(
                uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                collection_name=settings.MILVUS_COLLECTION
            )
        await self.vector_db.initialize()

    async def _init_embedding_model(self):
        """Initialize the embedding model with fallback."""
        try:
            self.embedding_model = GemmaEmbeddingModel(
                model_name=settings.EMBEDDING_MODEL,
                token=settings.HF_TOKEN
            )
            # If Gemma failed to load, use Mock
            if not self.embedding_model.model:
                logger.warning("Using Mock Embedding Model as fallback.")
                self.embedding_model = MockEmbeddingModel(dimension=settings.EMBEDDING_DIMENSION)
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = MockEmbeddingModel(dimension=settings.EMBEDDING_DIMENSION)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True
    )
    async def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Hybrid recommendation engine with retry logic for robustness.
        """
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"Processing recommendation request: {request}")
        
        results = []
        intent = None

        try:
            # 1. Handle Query-based search (Semantic Search)
            if request.query and self.vector_db and self.embedding_model:
                intent = await interpret_query(request.query)
                logger.info(f"Interpreted intent: {intent}")
                
                query_vector = await self.embedding_model.embed_text(intent.search_terms)
                
                # Prepare filters for vector DB if media_type is specified
                filters = request.filters or {}
                if request.media_type:
                    filters["media_type"] = request.media_type
                
                # Get more results than requested to allow for re-ranking/boosting
                try:
                    search_results = await self.vector_db.search(
                        query_vector=query_vector,
                        top_n=max(50, request.top_n),
                        filters=filters
                    )
                    
                    for hit in search_results:
                        media_data = self._get_media_by_id(hit['id'])
                        if media_data:
                            # Boost similarity score with a small fraction of the MAL/Media score
                            mal_score = media_data.get('score', 0)
                            if mal_score is None or pd.isna(mal_score):
                                mal_score = 5.0
                            
                            boosted_score = float(hit['score']) + (0.05 * (float(mal_score) / 10.0))
                            
                            results.append(MediaRecommendation(
                                **media_data,
                                similarity_score=boosted_score
                            ))
                except Exception as e:
                    logger.error(f"Vector search failed: {e}")
                    # Fallback to basic keyword search if possible, or just skip semantic
            elif request.query:
                logger.warning("Semantic search requested but vector DB/embedding model is unavailable.")

            # 2. Handle ID-based recommendations (Content-based)
            elif request.media_id:
                if self.content_filter:
                    logger.info(f"Getting content-based recommendations for ID: {request.media_id}")
                    cb_results = self.content_filter.recommend(request.media_id, top_n=request.top_n)
                    for _, row in cb_results.iterrows():
                        media_data = row.to_dict()
                        media_id = media_data.get('media_id', row.name)
                        media_data['media_id'] = int(media_id)
                        
                        # Clean NaN values safely
                        cleaned_data = {}
                        for k, v in media_data.items():
                            if isinstance(v, np.ndarray):
                                cleaned_data[k] = v.tolist()
                            elif isinstance(v, list):
                                cleaned_data[k] = v
                            elif pd.isna(v):
                                cleaned_data[k] = None
                            else:
                                cleaned_data[k] = v
                        
                        results.append(MediaRecommendation(
                            **cleaned_data,
                            similarity_score=float(row.get('similarity', 0.0))
                        ))

            # Sort and limit results
            results = sorted(results, key=lambda x: x.similarity_score, reverse=True)[:request.top_n]
            
            return RecommendationResponse(
                recommendations=results,
                metadata={
                    "intent": intent.model_dump() if intent else None,
                    "count": len(results),
                    "provider": self.provider
                }
            )
        except Exception as e:
            logger.error(f"Recommendation engine error: {e}", exc_info=True)
            raise

    async def get_trending(self, limit: int = 10, media_type: Optional[str] = None) -> List[MediaRecommendation]:
        """
        Get trending media based on score and popularity.
        """
        if not self.is_initialized:
            await self.initialize()

        if self.df is None or self.df.empty:
            return []

        # Filter by media_type if provided
        df_filtered = self.df
        if media_type:
            df_filtered = df_filtered[df_filtered['media_type'] == media_type]

        # Sort by score and popularity as a proxy for trending
        # We'll take the top items by score, but weighted by popularity if available
        sort_cols = []
        ascending = []
        
        if 'score' in df_filtered.columns:
            sort_cols.append('score')
            ascending.append(False)
        
        if 'popularity' in df_filtered.columns:
            sort_cols.append('popularity')
            ascending.append(True) # Lower popularity rank is better

        if not sort_cols:
            return []

        trending_df = df_filtered.sort_values(by=sort_cols, ascending=ascending).head(limit)
        
        results = []
        for _, row in trending_df.iterrows():
            results.append(self._row_to_recommendation(row))
            
        return results

    def _get_media_by_id(self, media_id: int) -> Optional[Dict[str, Any]]:
        """Helper to get full media details from dataframe."""
        if self.df is None:
            return None
        try:
            # Ensure media_id is scalar
            if isinstance(media_id, (list, np.ndarray)):
                if len(media_id) == 1:
                    media_id = media_id[0]
                else:
                    return None

            if media_id in self.df.index:
                row = self.df.loc[media_id]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
            else:
                matches = self.df[self.df['media_id'] == media_id]
                if matches.empty:
                    return None
                row = matches.iloc[0]
                
            data = row.to_dict()
            cleaned_data = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    cleaned_data[k] = v.tolist()
                elif isinstance(v, (list, tuple)):
                    cleaned_data[k] = list(v)
                elif pd.isna(v):
                    cleaned_data[k] = None
                else:
                    cleaned_data[k] = v
            
            # Ensure media_id is present
            if 'media_id' not in cleaned_data:
                cleaned_data['media_id'] = int(media_id)
            
            # Map studios/authors to metadata if they exist
            if 'studios' in cleaned_data and cleaned_data['studios']:
                cleaned_data['metadata'] = cleaned_data.get('metadata', {}) or {}
                cleaned_data['metadata']['studios'] = cleaned_data['studios']
                
            return cleaned_data
        except Exception as e:
            logger.warning(f"Error retrieving media {media_id}: {e}")
            return None

    async def ingest_data(self, batch_size: int = 100):
        """Ingest the loaded dataframe into the vector database."""
        if not self.is_initialized:
            await self.initialize()

        if self.df is None or self.df.empty:
            logger.warning("No data loaded to ingest.")
            return

        logger.info(f"Starting ingestion of {len(self.df)} records...")
        
        df_to_ingest = self.df.dropna(subset=['synopsis'])
        logger.info(f"Filtered to {len(df_to_ingest)} records with synopsis.")

        for i in range(0, len(df_to_ingest), batch_size):
            batch = df_to_ingest.iloc[i:i+batch_size]
            
            texts = []
            for _, row in batch.iterrows():
                # Combine title, english title, and synopsis for better semantic matching
                title_parts = [str(row['title'])]
                if 'alternative_titles_en' in row and pd.notnull(row['alternative_titles_en']):
                    title_parts.append(str(row['alternative_titles_en']))
                
                title_text = " / ".join(title_parts)
                texts.append(f"{title_text} ({row['media_type']}): {row['synopsis']}")
            
            embeddings = await self.embedding_model.embed_batch(texts)
            
            metadata = [
                {
                    "title": row['title'],
                    "genres": list(row['genres']) if isinstance(row['genres'], (list, np.ndarray)) else [],
                    "score": float(row['score']) if pd.notnull(row['score']) else 0.0,
                    "media_type": str(row['media_type']) if 'media_type' in row else "unknown"
                }
                for _, row in batch.iterrows()
            ]
            
            ids = batch['media_id'].astype(int).tolist()
            await self.vector_db.add_items(ids, embeddings, metadata)
            
            if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(df_to_ingest):
                logger.info(f"Ingested {min(i + batch_size, len(df_to_ingest))}/{len(df_to_ingest)} records")

        logger.info("Ingestion completed successfully.")

    async def search(self, query: str, limit: int = 10, offset: int = 0, genres: Optional[List[str]] = None, media_type: Optional[str] = None) -> List[MediaRecommendation]:
        """Simple semantic search wrapper with keyword fallback."""
        if not self.is_initialized:
            await self.initialize()

        filters = {}
        if genres:
            filters["genres"] = genres
        if media_type:
            filters["media_type"] = media_type
            
        # Try semantic search if available
        if self.vector_db and self.embedding_model:
            try:
                req = RecommendationRequest(query=query, top_n=limit + offset, filters=filters, media_type=media_type)
                resp = await self.get_recommendations(req)
                if resp.recommendations:
                    return resp.recommendations[offset:offset+limit]
            except Exception as e:
                logger.error(f"Semantic search error, falling back to keyword: {e}")

        # Fallback: Basic keyword search in dataframe
        if self.df is not None:
            query = query.lower()
            
            # Search in title, synopsis, and alternative titles
            mask = self.df['title'].str.lower().str.contains(query, na=False) | \
                    self.df['synopsis'].str.lower().str.contains(query, na=False)
            
            if 'alternative_titles_en' in self.df.columns:
                mask |= self.df['alternative_titles_en'].str.lower().str.contains(query, na=False)
            if 'alternative_titles_synonyms' in self.df.columns:
                # alternative_titles_synonyms might be a list or a string
                mask |= self.df['alternative_titles_synonyms'].apply(
                    lambda x: query in [str(s).lower() for s in x] if isinstance(x, (list, np.ndarray)) 
                    else (query in str(x).lower() if pd.notnull(x) else False)
                )
            
            if media_type:
                mask &= (self.df['media_type'] == media_type)
            
            if genres:
                # This is a bit slow for large DF, but works as fallback
                mask &= self.df['genres'].apply(lambda x: any(g in (x if isinstance(x, (list, np.ndarray)) else []) for g in genres))
            
            matches = self.df[mask].sort_values(by='score', ascending=False).iloc[offset:offset+limit]
            return [self._row_to_recommendation(row) for _, row in matches.iterrows()]
            
        return []

    async def check_health(self) -> Dict[str, Any]:
        """Check the health of the service and its dependencies."""
        health = {
            "status": "healthy",
            "initialized": self.is_initialized,
            "provider": self.provider,
            "vector_db": "unknown",
            "data_loaded": self.df is not None and not self.df.empty
        }
        
        if self.vector_db:
            try:
                count = await self.vector_db.count()
                health["vector_db"] = "connected"
                health["vector_count"] = count
            except Exception:
                health["vector_db"] = "disconnected"
                health["status"] = "degraded"
                
        return health

    async def get_by_id(self, media_id: int) -> Optional[MediaRecommendation]:
        if not self.is_initialized:
            await self.initialize()
            
        match = self.df[self.df['media_id'] == media_id]
        if match.empty:
            return None
        return self._row_to_recommendation(match.iloc[0])

    def get_all_genres(self, media_type: Optional[str] = None) -> List[str]:
        """Get a list of all unique genres."""
        if self.df is None or 'genres' not in self.df.columns:
            return []
        
        df = self.df
        if media_type:
            df = df[df['media_type'] == media_type]
            
        all_genres = set()
        for genres in df['genres'].dropna():
            if isinstance(genres, (list, np.ndarray)):
                all_genres.update(genres)
            elif isinstance(genres, str):
                try:
                    genres_list = eval(genres)
                    if isinstance(genres_list, list):
                        all_genres.update(genres_list)
                except:
                    pass
        
        return sorted(list(all_genres))
    
    async def get_cross_media_recommendations(
        self,
        media_id: str,
        target_media_types: List[str],
        top_n: int = 10
    ) -> List[MediaRecommendation]:
        """Get recommendations across different media types.
        
        Args:
            media_id: Source media ID (e.g., 'anime-123', 'manga-456')
            target_media_types: List of target media types to recommend
            top_n: Number of recommendations to return
            
        Returns:
            List of cross-media recommendations
        """
        if not self.is_initialized:
            await self.initialize()
        
        if self.df is None or self.df.empty:
            return []
        
        # Extract numeric ID from prefixed media_id
        try:
            if '-' in media_id:
                numeric_id = int(media_id.split('-')[1])
            else:
                numeric_id = int(media_id)
        except (ValueError, IndexError):
            logger.error(f"Invalid media_id format: {media_id}")
            return []
        
        # Get source media details
        source_media = self._get_media_by_id(numeric_id)
        if not source_media:
            logger.warning(f"Source media not found: {media_id}")
            return []
        
        # Use synopsis for semantic search if available
        if source_media.get('synopsis') and self.vector_db and self.embedding_model:
            try:
                # Create a search query from the source media
                query = f"{source_media['title']}: {source_media['synopsis']}"
                
                # Filter by target media types
                filters = {"media_type": target_media_types}
                
                req = RecommendationRequest(
                    query=query,
                    top_n=top_n * 2,  # Get more to filter
                    filters=filters
                )
                
                resp = await self.get_recommendations(req)
                
                # Filter out the source media itself
                results = [
                    rec for rec in resp.recommendations
                    if rec.media_id != numeric_id
                ][:top_n]
                
                return results
                
            except Exception as e:
                logger.error(f"Cross-media semantic search failed: {e}")
        
        # Fallback: Content-based filtering by genres
        if self.content_filter and source_media.get('genres'):
            try:
                # Filter by target media types
                df_filtered = self.df[self.df['media_type'].isin(target_media_types)]
                
                if df_filtered.empty:
                    return []
                
                # Find similar items based on genres
                source_genres = set(source_media['genres'])
                
                def genre_similarity(row):
                    if not isinstance(row.get('genres'), (list, np.ndarray)):
                        return 0.0
                    target_genres = set(row['genres'])
                    if not target_genres:
                        return 0.0
                    # Jaccard similarity
                    intersection = len(source_genres & target_genres)
                    union = len(source_genres | target_genres)
                    return intersection / union if union > 0 else 0.0
                
                df_filtered['_similarity'] = df_filtered.apply(genre_similarity, axis=1)
                df_filtered = df_filtered[df_filtered['_similarity'] > 0]
                df_filtered = df_filtered.sort_values('_similarity', ascending=False).head(top_n)
                
                results = []
                for _, row in df_filtered.iterrows():
                    results.append(self._row_to_recommendation(row, score=row['_similarity']))
                
                return results
                
            except Exception as e:
                logger.error(f"Cross-media content filtering failed: {e}")
        
        return []
    
    def filter_by_media_type(
        self,
        recommendations: List[MediaRecommendation],
        media_types: List[str]
    ) -> List[MediaRecommendation]:
        """Filter recommendations by media type.
        
        Args:
            recommendations: List of recommendations
            media_types: List of media types to include
            
        Returns:
            Filtered recommendations
        """
        return [
            rec for rec in recommendations
            if rec.media_type in media_types
        ]

    def _row_to_recommendation(self, row, score: float = 0.0) -> MediaRecommendation:
        def get_val(key, default=None):
            val = row.get(key)
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (list, tuple)):
                return list(val)
            if pd.isna(val):
                return default
            return val

        studios = get_val('studios')
        metadata = {}
        if studios is not None:
            if isinstance(studios, list):
                if len(studios) > 0:
                    metadata['studios'] = studios
            elif isinstance(studios, str) and studios:
                metadata['studios'] = [studios]

        return MediaRecommendation(
            media_id=int(get_val('media_id', 0)),
            title=str(get_val('title', 'Unknown')),
            synopsis=get_val('synopsis'),
            main_picture=get_val('main_picture') if isinstance(get_val('main_picture'), str) else None,
            score=float(get_val('score', 0.0)) if get_val('score') is not None else None,
            genres=get_val('genres', []),
            media_type=str(get_val('media_type', 'unknown')),
            status=get_val('status'),
            similarity_score=float(score),
            metadata=metadata
        )
