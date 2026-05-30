import { useState, useCallback, useRef } from "react";
import type { Anime } from "@/types/anime";

const RESULTS_PER_PAGE = 24;

export function useAnimeSearch() {
  const [results, setResults] = useState<Anime[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const selectedGenresRef = useRef<string[]>([]);

  const searchAnime = useCallback(
    async (query: string, genres: string[], newOffset = 0) => {
      if (!query.trim()) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      selectedGenresRef.current = genres;

      const isInitial = newOffset === 0;

      if (isInitial) {
        setLoading(true);
        setOffset(0);
        setResults([]);
        setHasMore(true);
      } else {
        setLoadingMore(true);
      }
      setError(null);

      try {
        const genreParams =
          genres.length > 0
            ? genres.map((g) => `&genres=${encodeURIComponent(g)}`).join("")
            : "";

        const res = await fetch(
          `/api/v1/anime/search?q=${encodeURIComponent(query)}&limit=${RESULTS_PER_PAGE}&offset=${newOffset}${genreParams}`,
          { signal: controller.signal }
        );
        if (!res.ok) throw new Error(`Search failed (${res.status})`);
        const data: Anime[] = await res.json();

        if (isInitial) {
          setResults(data);
        } else {
          setResults((prev) => [...prev, ...data]);
        }

        setHasMore(data.length === RESULTS_PER_PAGE);
        setOffset(newOffset);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        const message = err instanceof Error ? err.message : "Search failed";
        setError(message);
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
          setLoadingMore(false);
        }
      }
    },
    []
  );

  const loadMore = useCallback(
    (query: string, genres: string[]) => {
      if (loadingMore || !hasMore) return;
      searchAnime(query, genres, offset + RESULTS_PER_PAGE);
    },
    [loadingMore, hasMore, offset, searchAnime]
  );

  const clearResults = useCallback(() => {
    setResults([]);
    setHasMore(true);
    setOffset(0);
    setError(null);
  }, []);

  return {
    results,
    loading,
    loadingMore,
    hasMore,
    error,
    searchAnime,
    loadMore,
    clearResults,
  };
}
