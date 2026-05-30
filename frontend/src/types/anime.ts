export const MEDIA_TYPES = ["anime", "movies", "manga"] as const;
export type MediaType = (typeof MEDIA_TYPES)[number];

export interface Anime {
  anime_id: number;
  title: string;
  synopsis?: string;
  main_picture?: string;
  score?: number;
  genres: string[];
  studios?: string[];
  type?: string;
  episodes?: number;
  status?: string;
  media_type?: string;
  similarity_score?: number;
  metadata?: Record<string, unknown>;
}

export interface RecommendationResponse {
  recommendations: Anime[];
  total?: number;
  metadata?: Record<string, unknown>;
  query_intent?: {
    search_terms: string;
    genres: string[];
    min_score?: number;
    is_recommendation: boolean;
  };
}
