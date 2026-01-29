# Data Sources & API Strategy

This document outlines the selected data sources and APIs for expanding the Media Recommender system to support inclusive media types (Anime, Manga, Dramas, Movies, Books, Games, Music).

## 1. Anime (Existing)

* **Source:** [MyAnimeList (MAL) API](https://myanimelist.net/apiconfig/references/api/v2)
* **Status:** Implemented (`backend/src/anime_scraper.py`).
* **Pros:** Industry standard, rich metadata, user lists.
* **Cons:** Rate limits, strict attribution requirements.

## 2. Manga, Manhwa, Manhua

* **Primary Source:** [MangaDex API](https://api.mangadex.org/docs/)
* **Why:**
  * **Inclusive:** Covers Japanese Manga, Korean Manhwa, and Chinese Manhua.
  * **Open:** Free, public API with no strict rate limits (friendly policies).
  * **Content:** Access to covers, synopses, and chapter metadata.
* **Alternative:** Anilist GraphQL API (Good for aggregating metadata if MangaDex is missing info).

## 3. Asian Dramas (K-Drama, C-Drama, J-Drama)

* **Primary Source:** [The Movie Database (TMDB) API](https://developer.themoviedb.org/docs)
* **Why:**
  * **Stability:** High uptime, well-documented.
  * **Coverage:** Excellent coverage of popular Asian dramas with consistent metadata fields.

## 4. Western Movies & TV Series

* **Primary Source:** [The Movie Database (TMDB) API](https://developer.themoviedb.org/docs)
* **Why:** Industry standard for movie/TV metadata, posters, and cast info.
* **Alternative:** OMDb API (Good for simple queries, but TMDB is more comprehensive).

## 5. Books & Novels (Western)

* **Primary Source:** [Google Books API](https://developers.google.com/books)
* **Why:** Largest index of books, reliable search, ISBN matching.
* **Alternative:** [Open Library API](https://openlibrary.org/developers/api) (Open data, good for linking editions).

## 6. Asian Web Novels (Light Novels, Wuxia/Xianxia)

* **Primary Source:** [NovelUpdates](https://www.novelupdates.com/)
* **Strategy:** **Scraping Required.** No official API exists.
* **Implementation:** Use a headless browser or `BeautifulSoup` to scrape top lists and metadata. Respect `robots.txt` and implement caching to avoid IP bans.

## 7. Games

* **Primary Source:** [IGDB (Internet Game Database) API](https://api-docs.igdb.com/)
* **Why:** Owned by Twitch/Amazon. The gold standard for game metadata.
* **Auth:** Requires Twitch Developer account (free).
* **Alternative:** [RAWG.io API](https://rawg.io/apidocs) (Great visual data, free tier available).

## 8. Music

* **Primary Source:** [Spotify Web API](https://developer.spotify.com/documentation/web-api)
* **Why:** Rich audio features (danceability, energy, valence) perfect for semantic recommendations.
* **Alternative:** [MusicBrainz](https://musicbrainz.org/doc/MusicBrainz_API) (Open database, good for relationships/credits).

---

## Implementation Roadmap

1. **Unified Interface:** Create a `MediaClient` abstract base class in Python.
2. **Connectors:** Implement `MangaDexClient`, `TMDBClient`, `IGDBClient`, etc., inheriting from `MediaClient`.
3. **Data Ingestion:**
    * **Phase 1:** Manga (MangaDex) & Movies/Dramas (TMDB).
    * **Phase 2:** Games (IGDB) & Books (Google Books).
    * **Phase 3:** Music (Spotify) & Web Novels (NovelUpdates Scraper).
4. **Normalization:** Map all sources to the unified `MediaBase` model (Title, Synopsis, Image, Score, Genres).
