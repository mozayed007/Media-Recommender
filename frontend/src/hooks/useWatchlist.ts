import { useState, useEffect, useCallback } from "react";
import type { Anime } from "@/types/anime";

const STORAGE_KEY = "media-watchlist";

function isValidAnimeArray(value: unknown): value is Anime[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) =>
        typeof item === "object" &&
        item !== null &&
        typeof (item as Anime).anime_id === "number" &&
        typeof (item as Anime).title === "string"
    )
  );
}

function loadWatchlistFromStorage(): Anime[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (isValidAnimeArray(parsed)) {
        return parsed;
      }
    }
  } catch {
    // Corrupted data; start with empty list
  }
  return [];
}

export function useWatchlist() {
  const [watchlist, setWatchlist] = useState<Anime[]>(loadWatchlistFromStorage);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(watchlist));
  }, [watchlist]);

  const toggleWatchlist = useCallback((anime: Anime) => {
    setWatchlist((prev) => {
      const exists = prev.find((a) => a.anime_id === anime.anime_id);
      return exists
        ? prev.filter((a) => a.anime_id !== anime.anime_id)
        : [anime, ...prev];
    });
  }, []);

  return { watchlist, toggleWatchlist };
}
