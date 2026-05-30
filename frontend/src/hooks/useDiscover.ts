import { useState, useCallback } from "react";
import type { Anime } from "@/types/anime";

export interface DiscoverCategory {
  title: string;
  items: Anime[];
}

const CATEGORIES = [
  { title: "Action Masterpieces", query: "epic action anime with high stakes" },
  { title: "Emotional Journeys", query: "sad emotional anime that will make me cry" },
  { title: "Hidden Gems", query: "underrated anime with great story" },
] as const;

export function useDiscover() {
  const [discoverResults, setDiscoverResults] = useState<DiscoverCategory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDiscover = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const results = await Promise.all(
        CATEGORIES.map(async (cat) => {
          const res = await fetch(
            `/api/v1/anime/search?q=${encodeURIComponent(cat.query)}&limit=6`
          );
          if (!res.ok) throw new Error(`Failed to fetch ${cat.title}`);
          const data: Anime[] = await res.json();
          return { title: cat.title, items: data };
        })
      );
      setDiscoverResults(results);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load discover results";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { discoverResults, loading, error, fetchDiscover };
}
