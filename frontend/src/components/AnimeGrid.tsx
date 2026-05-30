import { AnimeCard } from "@/components/AnimeCard";
import type { Anime } from "@/types/anime";

interface AnimeGridProps {
  items: Anime[];
  gridKey: string;
  variant?: "default" | "recommendation";
  onAnimeClick: (anime: Anime) => void;
}

export function AnimeGrid({
  items,
  gridKey,
  variant = "default",
  onAnimeClick,
}: AnimeGridProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6">
      {items.map((anime, i) => (
        <AnimeCard
          key={`${gridKey}-${anime.anime_id}-${i}`}
          anime={anime}
          variant={variant}
          onClick={onAnimeClick}
        />
      ))}
    </div>
  );
}
