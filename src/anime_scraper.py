import os
from datetime import datetime
import tqdm
import time
import math
import json
import pandas as pd
import numpy as np
import logging
import argparse
from client import get_data
from pandas import Timedelta


def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def scrape_page(page, limit, anime_fields, endpoint, scraping_save_pages):
    logging.info(f"Started to scrape page {page}...")
    params = {'ranking_type': 'bypopularity', 'limit': limit, 'offset': page*limit, 'fields': ','.join(anime_fields)}
    data = get_data(endpoint, params)
    useful = [anime['node'] for anime in data['data']]
    file_path = os.path.join(scraping_save_pages, f'page{str(page).zfill(2)}.json')
    with open(file_path, 'w') as f:
        json.dump(useful, f, indent=4)
    logging.info(f"Finished scraping page {page}. Saved data to {file_path}.")

    
def merge_files(scraping_save_pages, output_file):
    print("Started to merge json files...")
    logging.info("Started to merge json files...")
    data = []
    for file_name in os.listdir(scraping_save_pages):
        file_path = os.path.join(scraping_save_pages, file_name)
        with open(file_path, 'r') as f:
            file = json.load(f)
            
        data.extend(file)
        print(f"Loaded data from {file_path}. Current total: {len(data)} animes.")
    print(f'There are {len(data)} animes in the json.')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Finished merging files. Saved data to {output_file}.")
    logging.info(f"Finished merging files. Saved data to {output_file}.")
    
def json_process(json_file, current_month_year):
    print(f"Started to process JSON file {json_file}...")
    logging.info(f"Started to process JSON file {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded data from {json_file}. Processing data...")
    anime = pd.json_normalize(data, sep='_')
    
    # Use Timestamps
    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d')
        except ValueError:
            try:
                return pd.to_datetime(date_str, format='%Y-%m')
            except ValueError:
                return pd.NaT

    anime['start_date'] = anime['start_date'].apply(parse_date)
    anime['end_date'] = anime['end_date'].apply(parse_date)

    # Rest of the function remains the same
    anime['num_episodes'] = anime['num_episodes'].replace(0, np.nan).astype('Int64')
    anime['popularity'] = anime['popularity'].replace(0, np.nan).astype('Int64')
    anime['rank'] = anime['rank'].replace(0, np.nan).astype('Int64')
    anime['mean'] = anime['mean'].replace(0, np.nan).astype('float64')
    anime['num_favorites'] = anime['num_favorites'].replace(0, np.nan).astype('Int64')
    anime['average_episode_duration'] = pd.to_timedelta(anime['average_episode_duration'].replace(0, np.nan), unit='s')
    anime['start_season_year'] = anime['start_season_year'].astype('Int64')
    anime['broadcast_start_time'] = pd.to_datetime(anime['broadcast_start_time']).dt.time
    anime['genres'] = anime['genres'].apply(lambda x: [dic['name'] for dic in x] if not x is np.nan else [])
    anime['studios'] = anime['studios'].apply(lambda x: [dic['name'] for dic in x] if not x is np.nan else [])
    anime['created_at'] = pd.to_datetime(anime['created_at']).dt.tz_convert(None)
    anime['updated_at'] = pd.to_datetime(anime['updated_at']).dt.tz_convert(None)
    anime['synopsis'] = anime['synopsis'].replace('', np.nan)
    anime['alternative_titles_en'] = anime['alternative_titles_en'].replace('', np.nan)
    anime['alternative_titles_ja'] = anime['alternative_titles_ja'].replace('', np.nan)
    
    columns_dtype_datetime = ['created_at', 'updated_at']
    for col in columns_dtype_datetime:
        anime[col] = pd.to_datetime(anime[col])
    columns_dtype_Int64 = ['num_episodes', 'popularity', 'rank', 'start_season_year']
    for col in columns_dtype_Int64:
        anime[col] = anime[col].astype('Int64')
    columns_dtype_list = ['genres', 'studios', 'alternative_titles_synonyms']
    for col in columns_dtype_list:
        anime[col] = anime[col].apply(lambda x: x.strip('[]').split(', ') if isinstance(x, str) else x)
    anime['broadcast_start_time'] = anime['broadcast_start_time'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time() if isinstance(x, str) else x)
    anime['average_episode_duration'] = anime['average_episode_duration'].apply(
        lambda x: x if isinstance(x, Timedelta) else pd.to_timedelta(float(x), unit='s') if pd.notnull(x) and is_float(x) else np.nan
    )
    
    # Rest of the function (ordering columns and saving data) remains the same
    order = ['id', 'title', 'media_type', 'mean', 'num_scoring_users',
            'status', 'num_episodes', 'start_date', 'end_date', 'source',
            'num_list_users', 'popularity', 'num_favorites', 'rank',
            'average_episode_duration', 'rating', 'start_season_year',
            'start_season_season', 'broadcast_day_of_the_week', 'broadcast_start_time',
            'genres', 'studios',
            'synopsis', 'nsfw', 'created_at', 'updated_at',
            'main_picture_medium', 'main_picture_large',
            'alternative_titles_en', 'alternative_titles_ja', 'alternative_titles_synonyms']
    
    print("Data processing completed. Reordering columns and saving data...")
    logging.info("Data processing completed. Reordering columns and saving data...")
    anime = anime.reindex(columns=order)
    csv_file = f'../data/raw/anime_{current_month_year}.csv'
    parquet_file = f'../data/raw/anime_{current_month_year}.parquet'
    anime.to_csv(csv_file, index=False)
    print(f"Saved data to {csv_file}.")
    logging.info(f"Saved data to {csv_file}.")
    anime.to_parquet(parquet_file, index=False)
    print(f"Saved data to {parquet_file}.")
    logging.info(f"Saved data to {parquet_file}.")

    
def main():
    # Get the current date and time
    now = datetime.now()
    # Format the current date and time as a string
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Set up logging
    logging.basicConfig(filename=f'../logs/anime_scraper_{now_str}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Scrape anime data.")
    parser.add_argument("--limit", type=int, default=500, help="The number of items to scrape per page.")
    parser.add_argument("--endpoint", type=str, default='/anime/ranking', help="The endpoint to scrape.")
    parser.add_argument("--scraping_save_pages", type=str, default='../data/raw/My Anime List Scrapping/data/data_tmp/anime_pages', help="The directory to save the scraped pages.")
    args = parser.parse_args()
    print('Starting the scraping process')
    logging.info('Starting the scraping process')
    print("\n" + "=" * 50)

    current_month_year = datetime.now().strftime('%b%y')
    scraping_save_pages = args.scraping_save_pages
    output_file = f'../data/raw/My Anime List Scrapping/data/data_tmp/anime_raw_{current_month_year}.json'
    if not os.path.exists(scraping_save_pages): # Create saving directory if it doesn't exist
        os.makedirs(scraping_save_pages)
        print(f"Created directory: {scraping_save_pages}")
        logging.info(f"Created directory: {scraping_save_pages}")
    else:
        print(f"Directory {scraping_save_pages} already exists")
        logging.info(f"Directory {scraping_save_pages} already exists")
    print("=" * 50 + "\n")
    if os.path.exists(output_file):
        print(f"Updated File: {output_file} already exists. Skipping scraping and merging.")
        logging.info(f"Updated File {output_file} already exists. Skipping scraping and merging.")
        json_process(output_file, current_month_year)
        print(f"JSON processing completed CSV and parquet files created for {current_month_year}.")
        logging.info(f"JSON processing completed CSV and parquet files created for {current_month_year}.")
        print("=" * 50 + "\n")
        print(" Exiting the scraping script....")
        logging.info("Exiting the scraping script....")
        print("=" * 50 + "\n")
        return
    else:
        endpoint = args.endpoint
        limit = args.limit
        anime_fields = ['id', 'title', 'main_picture', 'alternative_titles', 'start_date', 'end_date', 'synopsis', 'mean', 'rank', 'popularity',
                    'num_list_users', 'num_scoring_users', 'num_favorites', 'nsfw', 'genres', 'created_at', 'updated_at', 'media_type', 'status',
                    'num_episodes', 'start_season', 'broadcast', 'source', 'average_episode_duration', 'rating','studios']

        previous_total_anime = 35_000
        previous_last_page = math.ceil(previous_total_anime / limit) - 1
        print('Starting API calls...')
        data = get_data(endpoint, {'ranking_type': 'bypopularity', 'limit': limit, 'offset': previous_last_page*limit, 'fields': ','.join(anime_fields)})
        assert 'next' not in data['paging']
        last_page = previous_last_page
        print(f'The number of json pages is {last_page+1}')
        print("\n" + "=" * 50)
        for page in tqdm.trange(last_page+1):
            print(f"Scraping page {page}")
            scrape_page(page =page, limit=limit, anime_fields=anime_fields, endpoint=endpoint, scraping_save_pages=scraping_save_pages)
            time.sleep(1)
            
        print("Scraping process completed. Starting the merging process...")
        logging.info("Scraping process completed. Starting the merging process...")
        
        print("=" * 50 + "\n")
        merge_files(scraping_save_pages=scraping_save_pages, output_file=output_file)
        
        print("Merging process completed. Starting the JSON processing...")
        logging.info("Merging process completed. Starting the JSON processing...")
        print("\n" + "=" * 50)
        json_process(output_file,current_month_year)
        print(f"JSON processing completed CSV and parquet files created for {current_month_year}.")
        logging.info(f"JSON processing completed CSV and parquet files created for {current_month_year}.")
        print("=" * 50 + "\n")
        print(" Exiting the scraping script....")
        logging.info("Exiting the scraping script....")
        print("=" * 50 + "\n")
    
if __name__ == '__main__':
    main()
