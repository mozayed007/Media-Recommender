import Image from "next/image";
import { Anime } from "@/types/anime";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Star, Calendar, PlayCircle, Building2, Bookmark, BookmarkCheck } from "lucide-react";

interface AnimeDetailsModalProps {
  anime: Anime | null;
  isOpen: boolean;
  onClose: () => void;
  onGetRecommendations: (animeId: number) => void;
  onToggleWatchlist: (anime: Anime) => void;
  isInWatchlist: boolean;
}

export function AnimeDetailsModal({ 
  anime, 
  isOpen, 
  onClose, 
  onGetRecommendations,
  onToggleWatchlist,
  isInWatchlist
}: AnimeDetailsModalProps) {
  if (!anime) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl bg-gray-900 text-white border-gray-800">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold text-purple-400">{anime.title}</DialogTitle>
          <DialogDescription className="text-gray-400 flex items-center gap-4 mt-2">
            <span className="flex items-center gap-1"><Star className="w-4 h-4 text-yellow-500 fill-current" /> {anime.score || "N/A"}</span>
            <span className="flex items-center gap-1"><PlayCircle className="w-4 h-4" /> {anime.episodes || "?"} Episodes</span>
            <span className="flex items-center gap-1"><Calendar className="w-4 h-4" /> {anime.status || "Unknown"}</span>
          </DialogDescription>
        </DialogHeader>

        <div className="grid md:grid-cols-[250px_1fr] gap-6 mt-4">
          <div className="relative aspect-[3/4] rounded-lg overflow-hidden border border-gray-800">
            {anime.main_picture ? (
              <Image
                src={anime.main_picture}
                alt={anime.title}
                fill
                sizes="(max-width: 768px) 100vw, 250px"
                className="object-cover"
              />
            ) : (
              <div className="w-full h-full bg-gray-800 flex items-center justify-center">No Image</div>
            )}
          </div>

          <div className="space-y-6">
            <div>
              <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Synopsis</h4>
              <p className="text-gray-300 leading-relaxed text-sm max-h-[200px] overflow-y-auto pr-2 custom-scrollbar">
                {anime.synopsis || "No synopsis available."}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Genres</h4>
                <div className="flex flex-wrap gap-2">
                  {anime.genres.map((g, i) => (
                    <Badge key={`detail-genre-${g}-${i}`} variant="secondary" className="bg-purple-900/30 text-purple-300 border-purple-800/50">
                      {g}
                    </Badge>
                  ))}
                </div>
              </div>
              {anime.studios && anime.studios.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Studios</h4>
                  <div className="flex flex-wrap gap-2 text-gray-300 text-sm">
                    {anime.studios.map((s, i) => (
                      <span key={`detail-studio-${s}-${i}`} className="flex items-center gap-1">
                        <Building2 className="w-3 h-3" /> {s}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => {
                  onGetRecommendations(anime.anime_id);
                  onClose();
                }}
                className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold py-3 rounded-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-purple-900/20"
              >
                Find Similar
              </button>
              
              <button
                onClick={() => onToggleWatchlist(anime)}
                className={`px-4 rounded-lg border transition-all flex items-center justify-center gap-2 font-semibold ${
                  isInWatchlist 
                    ? "bg-purple-500/10 border-purple-500 text-purple-400 hover:bg-purple-500/20" 
                    : "border-gray-700 text-gray-400 hover:bg-white/5"
                }`}
                title={isInWatchlist ? "Remove from Watchlist" : "Add to Watchlist"}
              >
                {isInWatchlist ? <BookmarkCheck className="w-5 h-5" /> : <Bookmark className="w-5 h-5" />}
                <span className="hidden sm:inline">{isInWatchlist ? "Saved" : "Save"}</span>
              </button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
