import pandas as pd
import numpy as np
import datetime
from common_utils import base_directory
manga = pd.read_json('../data/raw/manga_mal.json')

# Usually no Duplicates, but can happen (it even happens in the website)
# ---------------------- BUT HERE THEY ARE REAL LOSSES!!!!!!!! ---------------------------------
old_size = manga.shape[0]
manga = manga.drop_duplicates(subset=['id']).reset_index(drop=True)
number_duplicates = old_size - manga.shape[0]
if number_duplicates:
    print('Duplicates:', number_duplicates)

# Shorter and better names, like MAL API
manga.rename(columns={'id': 'manga_id', 'media_type': 'type', 'mean': 'score', 'num_list_users': 'members', 'num_scoring_users': 'scored_by', \
    'num_favorites': 'favorites', 'num_volumes': 'volumes', 'num_chapters': 'chapters'}, inplace=True)

# Avoid false zeroes and unnecessary floats 
manga['volumes'] = manga['volumes'].replace(0, np.nan).astype('Int64')
manga['chapters'] = manga['chapters'].replace(0, np.nan).astype('Int64')

# Without adding False day 1 or False month January (i.e 2005 -> 2005-1-1)
manga['real_start_date'] = manga['start_date']
manga['real_end_date'] = manga['end_date']

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
manga['start_date'] = manga['start_date'].apply(process_date)
manga['end_date'] = manga['end_date'].apply(process_date)

# Then convert to datetime
manga['start_date'] = pd.to_datetime(manga['start_date'])
manga['end_date'] = pd.to_datetime(manga['end_date'])


# Use popularity=0 to detect 'pending approval' mangas
manga['approved'] = manga['popularity'] != 0

# Only keep names
manga['genres'] = manga['genres'].apply(lambda x: [dic['name'] for dic in x] if not x is np.nan else [])

genres = {'Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love',  'Comedy', 'Drama', 'Ecchi', 'Erotica', 'Fantasy',
'Girls Love', 'Gourmet', 'Hentai', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports', 'Supernatural', 'Suspense'}

themes = {'Adult Cast', 'Anthropomorphic', 'CGDCT', 'Childcare', 'Combat Sports', 'Crossdressing', 'Delinquents', 'Detective', 'Educational',
'Gag Humor', 'Gore', 'Harem', 'High Stakes Game', 'Historical', 'Idols (Female)', 'Idols (Male)', 'Isekai', 'Iyashikei', 'Love Polygon',
'Magical Sex Shift', 'Mahou Shoujo', 'Martial Arts', 'Mecha', 'Medical', 'Military', 'Music', 'Mythology', 'Organized Crime', 'Otaku Culture',
'Parody', 'Performing Arts', 'Pets', 'Psychological', 'Racing', 'Reincarnation', 'Reverse Harem', 'Romantic Subtext', 'Samurai', 'School',
'Showbiz', 'Space', 'Strategy Game', 'Super Power', 'Survival', 'Team Sports', 'Time Travel', 'Vampire', 'Video Game', 'Visual Arts',
'Workplace'} | {'Memoir', 'Villainess'}

demographics = {'Josei', 'Kids', 'Seinen', 'Shoujo', 'Shounen'}

# Split genres, themes and demographics
manga['themes'] = manga['genres'].apply(lambda x: [t for t in x if t in themes])
manga['demographics'] = manga['genres'].apply(lambda x: [t for t in x if t in demographics])
manga['genres'] = manga['genres'].apply(lambda x: [t for t in x if t in genres])

# Authors
def author_format(authors):
    if authors is np.nan:
        return []
    output = []
    for author in authors:
        output.append({'id': author['node']['id'], 'first_name': author['node']['first_name'], 'last_name': author['node']['last_name'], \
            'role': author['role']})
    return output
manga['authors']  = manga['authors'].apply(author_format)

# Mark R18+ Titles (not ranked)
manga['sfw'] = manga['genres'].apply(lambda x: 'Hentai' not in x and 'Erotica' not in x)

# Similar to the anime version, a lot of wrong labeled
manga.drop(columns=['nsfw'], inplace=True)

# MyAnimeList edits
for col in ['created_at', 'updated_at']:
    manga[col] = pd.to_datetime(manga[col])
    manga.loc[manga[col]=='1970-01-01 00:00:00+0000', col] = pd.NaT

# Looks like created_at it's not working??
assert all(manga['created_at'].isna())
manga.drop(columns=['created_at'], inplace=True)

# Make it manually
m = manga[manga['updated_at'].notna()].sort_values('updated_at')[['manga_id', 'updated_at']]
data = [m.iloc[0]]
for _, row in m.iterrows():
    if row['manga_id'] > data[-1]['manga_id']:
        data.append(row)
data.append({'manga_id': 2**63-1, 'updated_at': datetime.datetime.utcnow()})

created_at = []
manga.sort_values('manga_id', inplace=True)
pos = 0
for id in manga.manga_id:
    if id > data[pos]['manga_id']:
        pos += 1
    created_at.append(data[pos]['updated_at'])

manga['created_at_before'] = pd.to_datetime(created_at, utc=True)

# Avoid empty string
manga.loc[manga['synopsis'].isin(['', ' ', 'N/A', 'n/a']), 'synopsis'] = np.nan

# Simplify main picture
manga['main_picture'] = manga['main_picture'].str['large'].str.replace('api-', '')

# Normalize alternative titles
manga['title_english'] = manga['alternative_titles'].str['en'].replace('', np.nan)
manga['title_japanese'] = manga['alternative_titles'].str['ja'].replace('', np.nan)
manga['title_synonyms'] = manga['alternative_titles'].str['synonyms'].fillna('').apply(list)
manga.drop(columns=['alternative_titles'], inplace=True)

# Clean some string errors
for col in ['title', 'title_english', 'title_japanese']:
    manga[col] = manga[col].str.strip().str.replace('  ', ' ')
manga['title_synonyms'] = manga['title_synonyms'].apply(lambda x: [t.replace('  ', ' ') for t in x])

# Better order
order = ['manga_id', 'title', 'type', 'score', 'scored_by', 'status', 'volumes', 'chapters', 'start_date', 'end_date',
            'members', 'favorites', 'sfw', 'approved', 'created_at_before', 'updated_at', 'real_start_date', 'real_end_date',
            'genres', 'themes', 'demographics', 'authors', 'synopsis', 'main_picture', 'title_english', 'title_japanese', 'title_synonyms']

deleted = ['rank', 'popularity', 'nsfw']

missing = ['background', 'serializations', 'url']

manga = manga[order]

# Sort by Top Manga
manga['tmp'] = manga['score'].rank(ascending=False) + manga['scored_by'].rank(ascending=False)
manga = manga.sort_values(['tmp', 'members', 'favorites', 'manga_id'], \
    ascending=[True, False, False, True]).reset_index(drop=True)
manga.drop(columns=['tmp'], inplace=True)
# Get the current month name
current_month = datetime.datetime.now().strftime('%B')
# Save to csv
manga.to_csv(f'..{base_directory}/manga_mal_{current_month}.csv', index=False)