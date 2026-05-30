import { Anime } from "@/types/anime";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Star, Info } from "lucide-react";
import Image from "next/image";

interface AnimeCardProps {
  anime: Anime;
  onClick?: (anime: Anime) => void;
  variant?: "default" | "recommendation";
}

export function AnimeCard({ anime, onClick, variant = "default" }: AnimeCardProps) {
  const isRecommendation = variant === "recommendation";

  return (
    <Card 
      className={`group overflow-hidden cursor-pointer transition-all duration-300 hover:ring-2 ${
        isRecommendation ? "hover:ring-pink-500 border-pink-900/20" : "hover:ring-purple-500 border-purple-900/20"
      } bg-gray-900/50 backdrop-blur-sm border`}
      onClick={() => onClick?.(anime)}
    >
      <div className="relative aspect-[3/4] overflow-hidden">
        {anime.main_picture ? (
          <Image
            src={anime.main_picture}
            alt={anime.title}
            fill
            sizes="(max-width: 768px) 50vw, (max-width: 1200px) 33vw, 20vw"
            className="object-cover transition-transform duration-500 group-hover:scale-110"
          />
        ) : (
          <div className="w-full h-full bg-gray-800 flex items-center justify-center">
            <span className="text-gray-500 text-xs">No Image</span>
          </div>
        )}
        
        {/* Similarity Score Overlay */}
        {anime.similarity_score && (
          <div className="absolute top-2 right-2 z-10">
            <Badge className={`${isRecommendation ? "bg-pink-600" : "bg-purple-600"} hover:bg-opacity-100`}>
              {Math.round(anime.similarity_score * 100)}% Match
            </Badge>
          </div>
        )}

        {/* Hover Overlay */}
        <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col items-center justify-center p-4 text-center">
          <Info className={`w-8 h-8 mb-2 ${isRecommendation ? "text-pink-400" : "text-purple-400"}`} />
          <p className="text-xs text-gray-200 line-clamp-4">{anime.synopsis}</p>
          <span className="mt-4 text-sm font-semibold">View Details</span>
        </div>
      </div>

      <CardContent className="p-3">
        <h3 className="font-bold text-sm truncate text-gray-100 group-hover:text-white transition-colors">
          {anime.title}
        </h3>
        <div className="flex items-center gap-2 mt-1.5">
          <div className="flex items-center text-yellow-500">
            <Star className="w-3 h-3 fill-current" />
            <span className="text-[11px] font-medium ml-1">{anime.score || "N/A"}</span>
          </div>
          {anime.type && (
            <Badge variant="outline" className="text-[9px] py-0 px-1.5 border-gray-700 text-gray-400">
              {anime.type}
            </Badge>
          )}
        </div>
      </CardContent>

      <CardFooter className="p-3 pt-0 flex flex-wrap gap-1">
        {anime.genres.slice(0, 2).map((genre, i) => (
          <Badge 
            key={`${anime.anime_id}-genre-${genre}-${i}`} 
            variant="secondary" 
            className="text-[9px] py-0 px-1.5 bg-gray-800 text-gray-400 hover:text-gray-200"
          >
            {genre}
          </Badge>
        ))}
      </CardFooter>
    </Card>
  );
}
