"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Search,
  Sparkles,
  TrendingUp,
  Loader2,
  X,
  Bookmark,
  ArrowUp,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import type { Anime, MediaType } from "@/types/anime";
import { AnimeDetailsModal } from "@/components/AnimeDetailsModal";
import { AnimeGrid } from "@/components/AnimeGrid";
import { SectionHeader } from "@/components/SectionHeader";
import { ErrorBanner } from "@/components/ErrorBanner";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { useAnimeSearch } from "@/hooks/useAnimeSearch";
import { useWatchlist } from "@/hooks/useWatchlist";
import { useDiscover } from "@/hooks/useDiscover";

const MEDIA_TYPE_OPTIONS: { id: MediaType; label: string; icon: typeof Sparkles }[] = [
  { id: "anime", label: "Anime", icon: Sparkles },
  { id: "movies", label: "Movies", icon: Search },
  { id: "manga", label: "Manga", icon: Bookmark },
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [mediaType, setMediaType] = useState<MediaType>("anime");
  const [showWatchlist, setShowWatchlist] = useState(false);
  const [showDiscover, setShowDiscover] = useState(false);
  const [selectedAnime, setSelectedAnime] = useState<Anime | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [showBackToTop, setShowBackToTop] = useState(false);
  const [backendError, setBackendError] = useState(false);
  const [genres, setGenres] = useState<string[]>([]);
  const [trending, setTrending] = useState<Anime[]>([]);
  const [recommendations, setRecommendations] = useState<Anime[]>([]);
  const [recommendationsError, setRecommendationsError] = useState<string | null>(null);

  const {
    results,
    loading: searchLoading,
    loadingMore,
    hasMore,
    error: searchError,
    searchAnime,
    loadMore,
    clearResults,
  } = useAnimeSearch();

  const { watchlist, toggleWatchlist } = useWatchlist();

  const {
    discoverResults,
    loading: discoverLoading,
    error: discoverError,
    fetchDiscover,
  } = useDiscover();

  const loading = searchLoading || discoverLoading;

  // Fetch genres and trending on mount / mediaType change
  useEffect(() => {
    const ac = new AbortController();

    async function fetchGenres() {
      if (mediaType !== "anime") {
        setGenres([]);
        return;
      }
      try {
        const res = await fetch("/api/v1/anime/genres", { signal: ac.signal });
        if (!res.ok) throw new Error("Failed to fetch genres");
        setGenres(await res.json());
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        console.error("Error fetching genres:", err);
      }
    }

    async function fetchTrending() {
      try {
        const res = await fetch("/api/v1/anime/trending?limit=12", {
          signal: ac.signal,
        });
        if (!res.ok) throw new Error("Failed to fetch trending");
        setTrending(await res.json());
        setBackendError(false);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        console.error("Error fetching trending:", err);
        setBackendError(true);
      }
    }

    fetchGenres();
    fetchTrending();
    return () => ac.abort();
  }, [mediaType]);

  // Scroll listener for back-to-top
  useEffect(() => {
    const handleScroll = () => setShowBackToTop(window.scrollY > 400);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleAnimeClick = useCallback((anime: Anime) => {
    setSelectedAnime(anime);
    setIsModalOpen(true);
  }, []);

  const getRecommendations = useCallback(async (animeId: number) => {
    setRecommendationsError(null);
    try {
      const res = await fetch("/api/v1/anime/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ anime_id: animeId, top_n: 12 }),
      });
      if (!res.ok) throw new Error("Recommendation failed");
      const data = await res.json();
      setRecommendations(data.recommendations);
      setTimeout(() => {
        document
          .getElementById("recommendations")
          ?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to get recommendations";
      setRecommendationsError(message);
    }
  }, []);

  const handleSearch = useCallback(() => {
    setShowWatchlist(false);
    setShowDiscover(false);
    searchAnime(query, selectedGenres);
  }, [query, selectedGenres, searchAnime]);

  const handleToggleGenre = useCallback(
    (genre: string) => {
      setSelectedGenres((prev) =>
        prev.includes(genre)
          ? prev.filter((g) => g !== genre)
          : [...prev, genre]
      );
    },
    []
  );

  const handleDiscover = useCallback(() => {
    setShowDiscover(true);
    setShowWatchlist(false);
    clearResults();
    setRecommendations([]);
    fetchDiscover();
  }, [fetchDiscover, clearResults]);

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-gray-100 font-sans selection:bg-purple-500/30">
      {backendError && (
        <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 text-center text-amber-400 text-sm">
          Backend server seems to be starting up or unreachable. Please wait a
          few seconds and
          <button
            onClick={() => window.location.reload()}
            className="ml-2 underline font-bold hover:text-amber-300"
          >
            refresh
          </button>
          .
        </div>
      )}

      {/* Hero Section */}
      <div className="relative overflow-hidden border-b border-white/5 bg-gradient-to-b from-purple-900/10 to-transparent pt-16 pb-24">
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-10" />
        <div className="container max-w-6xl mx-auto px-6 relative z-10">
          <div className="flex flex-col items-center text-center space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs font-medium animate-pulse">
              <Sparkles className="w-3 h-3" />
              RAG-Powered AI Recommendations
            </div>

            <div className="flex justify-center gap-2 mb-4">
              {MEDIA_TYPE_OPTIONS.map((type) => (
                <button
                  key={type.id}
                  onClick={() => setMediaType(type.id)}
                  className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all flex items-center gap-2 ${
                    mediaType === type.id
                      ? "bg-white text-black shadow-lg"
                      : "text-gray-500 hover:text-white bg-white/5 hover:bg-white/10"
                  }`}
                >
                  <type.icon className="w-3.5 h-3.5" />
                  {type.label}
                  {type.id !== "anime" && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 font-bold uppercase tracking-wider">
                      Soon
                    </span>
                  )}
                </button>
              ))}
            </div>

            <h1 className="text-6xl md:text-7xl font-black tracking-tight bg-gradient-to-r from-white via-purple-200 to-gray-500 bg-clip-text text-transparent">
              Discover Your Next <br /> Favorite Media
            </h1>
            <p className="text-gray-400 text-lg md:text-xl max-w-2xl font-light">
              Starting with the best of Anime, we&apos;re building the ultimate
              media recommendation engine. Search semantically or find gems
              similar to what you already love.
            </p>

            <div className="w-full max-w-3xl flex gap-3 p-2 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl shadow-purple-500/5 focus-within:border-purple-500/50 transition-all">
              <div className="relative flex-1">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                  placeholder="Ask for something... (e.g. 'A high school romance with a sad ending')"
                  className="w-full h-14 pl-12 bg-transparent border-none text-lg placeholder:text-gray-600 focus-visible:ring-0 focus-visible:ring-offset-0"
                />
              </div>
              <Button
                onClick={handleSearch}
                disabled={loading}
                className="h-14 px-8 bg-purple-600 hover:bg-purple-700 text-white rounded-xl transition-all shadow-lg shadow-purple-900/20 active:scale-95"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Search className="w-5 h-5" />
                )}
                <span className="ml-2 font-semibold">Search</span>
              </Button>
            </div>

            {/* Watchlist & Discover Toggles */}
            <div className="flex justify-center gap-4">
              <Button
                variant="ghost"
                onClick={handleDiscover}
                className={`flex items-center gap-2 transition-all ${
                  showDiscover
                    ? "text-purple-400 bg-purple-500/10"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <Sparkles
                  className={`w-5 h-5 ${showDiscover ? "fill-current" : ""}`}
                />
                Discover
              </Button>
              <Button
                variant="ghost"
                onClick={() => {
                  setShowWatchlist((prev) => !prev);
                  setShowDiscover(false);
                }}
                className={`flex items-center gap-2 transition-all ${
                  showWatchlist
                    ? "text-purple-400 bg-purple-500/10"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <Bookmark
                  className={`w-5 h-5 ${showWatchlist ? "fill-current" : ""}`}
                />
                My Watchlist ({watchlist.length})
              </Button>
            </div>

            <div className="flex flex-wrap justify-center gap-2 max-w-2xl">
              {genres.slice(0, 15).map((genre, i) => (
                <Badge
                  key={`genre-${genre}-${i}`}
                  variant={selectedGenres.includes(genre) ? "default" : "outline"}
                  className={`cursor-pointer transition-all ${
                    selectedGenres.includes(genre)
                      ? "bg-purple-600 hover:bg-purple-700 border-transparent"
                      : "hover:bg-white/5 border-white/10 text-gray-400"
                  }`}
                  onClick={() => handleToggleGenre(genre)}
                >
                  {genre}
                </Badge>
              ))}
              {genres.length > 15 && (
                <Badge
                  variant="outline"
                  className="border-white/10 text-gray-500 italic"
                >
                  +{genres.length - 15} more
                </Badge>
              )}
            </div>
          </div>
        </div>
      </div>

      <main className="container max-w-7xl mx-auto px-6 py-16 space-y-24">
        {/* Global search error */}
        {searchError && (
          <ErrorBanner message={searchError} onDismiss={clearResults} />
        )}
        {discoverError && (
          <ErrorBanner message={discoverError} onDismiss={() => {}} />
        )}
        {recommendationsError && (
          <ErrorBanner
            message={recommendationsError}
            onDismiss={() => setRecommendationsError(null)}
          />
        )}

        <AnimatePresence mode="wait">
          {/* Watchlist Section */}
          {showWatchlist && (
            <motion.section
              key="watchlist"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              <SectionHeader
                icon={Bookmark}
                iconColor="text-pink-400"
                iconBg="bg-pink-500/10"
                title="My Watchlist"
                titleColor="text-pink-400"
                subtitle="Your saved media to watch later"
                action={
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowWatchlist(false)}
                    className="text-gray-500 hover:text-white"
                  >
                    <X className="w-4 h-4 mr-2" /> Close
                  </Button>
                }
              />

              {watchlist.length > 0 ? (
                <AnimeGrid
                  items={watchlist}
                  gridKey="watchlist"
                  onAnimeClick={handleAnimeClick}
                />
              ) : (
                <div className="flex flex-col items-center justify-center py-20 text-center space-y-4 bg-white/5 rounded-3xl border border-dashed border-white/10">
                  <Bookmark className="w-12 h-12 text-gray-700" />
                  <div className="space-y-1">
                    <p className="text-xl font-medium text-gray-400">
                      Your watchlist is empty
                    </p>
                    <p className="text-gray-500">
                      Save anime you want to watch later to see them here.
                    </p>
                  </div>
                </div>
              )}
            </motion.section>
          )}

          {/* Discover Section */}
          {showDiscover && !discoverLoading && (
            <motion.section
              key="discover"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-16"
            >
              {discoverResults.map((category, catIdx) => (
                <div
                  key={`discover-cat-${category.title}-${catIdx}`}
                  className="space-y-8"
                >
                  <SectionHeader
                    icon={Sparkles}
                    title={category.title}
                    subtitle="Hand-picked by our semantic AI"
                  />
                  <AnimeGrid
                    items={category.items}
                    gridKey={`discover-${category.title}`}
                    onAnimeClick={handleAnimeClick}
                  />
                </div>
              ))}
            </motion.section>
          )}

          {/* Search Results */}
          {results.length > 0 && !showWatchlist && !showDiscover && (
            <motion.section
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              <SectionHeader
                icon={Search}
                title="Search Results"
                subtitle={`Found ${results.length} titles matching your query`}
                action={
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearResults}
                    className="text-gray-500 hover:text-white"
                  >
                    <X className="w-4 h-4 mr-2" /> Clear Results
                  </Button>
                }
              />

              <AnimeGrid
                items={results}
                gridKey="search-result"
                onAnimeClick={handleAnimeClick}
              />

              {hasMore && (
                <div className="flex justify-center pt-8">
                  <Button
                    variant="outline"
                    size="lg"
                    onClick={() => loadMore(query, selectedGenres)}
                    disabled={loadingMore}
                    className="bg-white/5 border-white/10 hover:bg-white/10 text-white min-w-[200px]"
                  >
                    {loadingMore ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Loading more...
                      </>
                    ) : (
                      "Load More Results"
                    )}
                  </Button>
                </div>
              )}
            </motion.section>
          )}

          {/* Recommendations Section */}
          {recommendations.length > 0 && !showWatchlist && !showDiscover && (
            <motion.section
              key="recommendations"
              id="recommendations"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              <SectionHeader
                icon={Sparkles}
                iconColor="text-pink-400"
                iconBg="bg-pink-500/10"
                title="Recommended for You"
                titleColor="text-pink-400"
                subtitle="Personalized picks based on your interest"
              />
              <AnimeGrid
                items={recommendations}
                gridKey="recommendation"
                variant="recommendation"
                onAnimeClick={handleAnimeClick}
              />
            </motion.section>
          )}

          {/* Trending Section */}
          {!results.length &&
            !recommendations.length &&
            trending.length > 0 &&
            !loading &&
            !showWatchlist &&
            !showDiscover && (
              <motion.section
                key="trending"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-8"
              >
                <SectionHeader
                  icon={TrendingUp}
                  title="Top Rated Anime"
                  subtitle="Global favorites and all-time classics"
                />
                <AnimeGrid
                  items={trending}
                  gridKey="trending"
                  onAnimeClick={handleAnimeClick}
                />
              </motion.section>
            )}
        </AnimatePresence>

        {/* Empty State */}
        {!results.length &&
          !recommendations.length &&
          !trending.length &&
          !loading &&
          !showDiscover &&
          !showWatchlist && (
            <div className="flex flex-col items-center justify-center py-20 text-center space-y-8 opacity-40 grayscale hover:opacity-100 hover:grayscale-0 transition-all duration-700">
              <div className="relative">
                <TrendingUp className="w-24 h-24 text-purple-500/20" />
                <Sparkles className="absolute -top-2 -right-2 w-8 h-8 text-pink-500/40 animate-pulse" />
              </div>
              <div className="space-y-2">
                <h3 className="text-2xl font-bold text-gray-400">
                  No results yet
                </h3>
                <p className="text-gray-500 max-w-sm mx-auto">
                  Try searching for specific genres, themes, or even natural
                  language descriptions of what you&apos;re in the mood for.
                </p>
              </div>
            </div>
          )}

        {/* Loading State */}
        {loading && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6 pt-16">
            {[...Array(12)].map((_, i) => (
              <div key={i} className="space-y-4">
                <Skeleton className="aspect-[3/4] w-full rounded-xl bg-white/5" />
                <div className="space-y-2">
                  <Skeleton className="h-4 w-3/4 bg-white/5" />
                  <Skeleton className="h-3 w-1/2 bg-white/5" />
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Back to Top */}
      <AnimatePresence>
        {showBackToTop && (
          <motion.button
            key="back-to-top"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5 }}
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className="fixed bottom-8 right-8 z-50 p-4 rounded-full bg-purple-600 text-white shadow-xl shadow-purple-900/40 hover:bg-purple-700 transition-colors"
          >
            <ArrowUp className="w-6 h-6" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Footer */}
      <footer className="border-t border-white/5 py-12 bg-black/50 backdrop-blur-sm mt-24">
        <div className="container max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="space-y-2">
            <h4 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
              Media Recommender
            </h4>
            <p className="text-gray-500 text-sm">
              © 2026 Powered by Gemini &amp; Vector Search
            </p>
          </div>
          <div className="flex gap-8 text-sm text-gray-500">
            <a href="#" className="hover:text-purple-400 transition-colors">
              Documentation
            </a>
            <a href="#" className="hover:text-purple-400 transition-colors">
              API Status
            </a>
            <a href="#" className="hover:text-purple-400 transition-colors">
              Privacy Policy
            </a>
          </div>
        </div>
      </footer>

      {/* Modal */}
      <AnimeDetailsModal
        anime={selectedAnime}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onGetRecommendations={getRecommendations}
        onToggleWatchlist={toggleWatchlist}
        isInWatchlist={
          selectedAnime
            ? watchlist.some((a) => a.anime_id === selectedAnime.anime_id)
            : false
        }
      />
    </div>
  );
}
