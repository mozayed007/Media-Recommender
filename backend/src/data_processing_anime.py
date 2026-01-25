import pandas as pd
import numpy as np
from datetime import datetime

anime = pd.read_json('../data/raw/anime_mal.json')

# Usually no Duplicates, but can happen (it even happens in the website)
old_size = anime.shape[0]
anime = anime.drop_duplicates(subset=['id']).reset_index(drop=True)
number_duplicates = old_size - anime.shape[0]
if number_duplicates:
    print('Duplicates:', number_duplicates)

# Shorter and better names, like the website
anime.rename(columns={'id': 'anime_id', 'media_type': 'type', 'mean': 'score', 'num_list_users': 'members', 'num_scoring_users': 'scored_by', \
    'num_favorites': 'favorites', 'average_episode_duration': 'episode_duration', 'num_episodes': 'episodes'}, inplace=True)

# Avoid 'Unknown' string
anime['type'] = anime['type'].replace('unknown', np.nan)

# Avoid false zeroes and unnecessary floats
anime['episodes'] = anime['episodes'].replace(0, np.nan).astype('Int64')

# Without adding False day 1 or False month January (i.e 2005 -> 2005-1-1)
anime['real_start_date'] = anime['start_date']
anime['real_end_date'] = anime['end_date']
# Helper function to correctly process dates
def process_date(x):
    # Check for string type
    if isinstance(x, str):
        parts = x.split('-')
        # If date string doesn't have a month or day component
        if len(parts) < 3:
            # If it doesn't have a month component, add '01'
            if len(parts) == 1:
                return f"{x}-01-01"
            # If it has a month component but no day, add '01'
            elif len(parts) == 2:
                return f"{x}-01"
        else:
            return x
    else:
        # Handle NaN/float values
        return np.nan

# Apply the function to 'start_date' and 'end_date'
anime['start_date'] = anime['start_date'].apply(process_date)
anime['end_date'] = anime['end_date'].apply(process_date)

# Then convert to datetime
anime['start_date'] = pd.to_datetime(anime['start_date'])
anime['end_date'] = pd.to_datetime(anime['end_date'])


# Use Timedelta
anime['episode_duration'] = pd.to_timedelta(anime['episode_duration'].replace(0, np.nan), unit='s')
anime['total_duration'] = anime.apply(lambda x: x['episode_duration'] * x['episodes'] if not pd.isna(x['episodes']) else np.nan, axis=1)

# Use popularity=0 to detect 'pending approval' animes
anime['approved'] = anime['popularity'] != 0

#  Drop rank and popularity, as they sort equal score / members alphabetically...
anime.drop(columns=['rank', 'popularity'], inplace=True)

# MyAnimeList edits
anime['created_at'] = pd.to_datetime(anime['created_at'])
anime['updated_at'] = pd.to_datetime(anime['updated_at'])

# Normalize start season
anime['start_year'] = anime['start_season'].str['year'].astype('Int64')
anime['start_season'] = anime['start_season'].str['season']

# Avoid empty synopsis
old_default_synopsis = 'No synopsis has been added for this series yet.\n\nClick here to update this information.'
anime['synopsis'] = anime['synopsis'].replace('', np.nan).replace(old_default_synopsis, np.nan)

# Simplify main picture
anime['main_picture'] = anime['main_picture'].str['large'].str.replace('api-', '')

# Normalize broadcast
anime['broadcast_day'] = anime['broadcast'].str['day_of_the_week']
# # Convert 'broadcast_time' to time format
# anime['broadcast_time'] = pd.to_datetime(anime['broadcast'].str['start_time']).dt.time
anime.drop(columns=['broadcast'], inplace=True)

# Only keep names
anime['genres'] = anime['genres'].apply(lambda x: [dic['name'] for dic in x] if not x is np.nan else [])
anime['studios'] = anime['studios'].apply(lambda x: [dic['name'] for dic in x] if not x is np.nan else [])

genres = {'Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love',  'Comedy', 'Drama', 'Ecchi', 'Erotica', 'Fantasy',
'Girls Love', 'Gourmet', 'Hentai', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports', 'Supernatural', 'Suspense'}

themes = {'Adult Cast', 'Anthropomorphic', 'CGDCT', 'Childcare', 'Combat Sports', 'Crossdressing', 'Delinquents', 'Detective', 'Educational',
'Gag Humor', 'Gore', 'Harem', 'High Stakes Game', 'Historical', 'Idols (Female)', 'Idols (Male)', 'Isekai', 'Iyashikei', 'Love Polygon',
'Magical Sex Shift', 'Mahou Shoujo', 'Martial Arts', 'Mecha', 'Medical', 'Military', 'Music', 'Mythology', 'Organized Crime', 'Otaku Culture',
'Parody', 'Performing Arts', 'Pets', 'Psychological', 'Racing', 'Reincarnation', 'Reverse Harem', 'Romantic Subtext', 'Samurai', 'School',
'Showbiz', 'Space', 'Strategy Game', 'Super Power', 'Survival', 'Team Sports', 'Time Travel', 'Vampire', 'Video Game', 'Visual Arts', 'Workplace'}

demographics = {'Josei', 'Kids', 'Seinen', 'Shoujo', 'Shounen'}

# Split genres, themes and demographics
anime['themes'] = anime['genres'].apply(lambda x: [t for t in x if t in themes])
anime['demographics'] = anime['genres'].apply(lambda x: [t for t in x if t in demographics])
anime['genres'] = anime['genres'].apply(lambda x: [t for t in x if t in genres])

# Mark R18+ Titles (not ranked)
anime['sfw'] = anime['genres'].apply(lambda x: 'Hentai' not in x and 'Erotica' not in x)

# nsfw is much more restrictive. But on 2022-9-22 it was deprecated and it's not used anymore. It has a lot of false positives, and is no
# longer updated, so the new definition is simply better, nudity it's already marked with r+. Only rember to mark it when requesting lists
anime.drop(columns=['nsfw'], inplace=True)

# Alternative titles
anime['title_english'] = anime['alternative_titles'].str['en'].replace('', np.nan)
anime['title_japanese'] = anime['alternative_titles'].str['ja'].replace('', np.nan)
anime['title_synonyms'] = anime['alternative_titles'].str['synonyms']
anime.drop(columns=['alternative_titles'], inplace=True)

# Avoid double spaces, which don't appear on the website
for col in ['title', 'title_english', 'title_japanese']:
    anime[col] = anime[col].str.replace('  ', ' ')
anime['title_synonyms'] = anime['title_synonyms'].apply(lambda x: [t.replace('  ', ' ') for t in x])

# Better order
order = ['anime_id', 'title', 'type', 'score', 'scored_by', 'status', 'episodes', 'start_date', 'end_date', 'source',
        'members', 'favorites', 'episode_duration', 'total_duration', 'rating', 'sfw', 'approved', 'created_at', 'updated_at',
        'start_year', 'start_season', 'real_start_date', 'real_end_date', 'broadcast_day',
        'genres', 'themes', 'demographics', 'studios', 'synopsis', 'main_picture', 'title_english', 'title_japanese', 'title_synonyms']

deleted = ['rank', 'popularity', 'nsfw']

missing = ['producers', 'licensors', 'background', 'url', 'trailer_url']

anime = anime[order]

# Sort by Top Anime
anime['tmp'] = anime['score'].rank(ascending=False) + anime['scored_by'].rank(ascending=False)
anime = anime.sort_values(['tmp', 'members', 'favorites', 'anime_id'], \
    ascending=[True, False, False, True]).reset_index(drop=True)
anime.drop(columns=['tmp'], inplace=True)

# Save to parquet and csv
base_directory = '/data/processed'
current_month_year = datetime.now().strftime('%b%y')
anime.to_csv(f'..{base_directory}/anime_mal_{current_month_year}.csv', index=False)
anime.to_parquet(f'..{base_directory}/anime_mal_{current_month_year}.parquet', index=False)