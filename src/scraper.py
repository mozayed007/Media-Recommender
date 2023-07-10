import os
import datetime
import tqdm
import time
import math
import json
from client import get_data
from common_utils import keys, anime_keys, manga_keys, known_fails, base_directory, merge_anime

def scrape_ranking_page(database, ranking_type, page, fields, tmp_directory, length):
    params = {'ranking_type': ranking_type, 'limit': 500, 'offset': page*500, 'fields': fields}
    try:
        data = get_data(f'/{database}/ranking', params)
    except:
        data = manga_crash(f'/{database}/ranking', params)

    useful = [anime['node'] for anime in data['data']]
    # Saves the file in your /tmp_{database}_mal folder
    with open(f'{tmp_directory}page{str(page).zfill(length)}.json', 'w') as f:
        json.dump(useful, f)

def scrape_ranking(database='anime', ranking_type='favorite'):

    # Check if keys exists in the keys dictionary
    if database not in keys:
        print(f"No keys found for the database: {database}")
        return

    # This is the directory where the merged file will be saved
    save_directory = r'../data/raw/'
    save_file_path = f'{save_directory}{database}_mal.json'

    # Check if save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # This is your temporary directory
    tmp_directory = f'../data/raw/tmp_{database}_mal/'

    # Check if temp directory exists
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    fields = ','.join(keys[database])
    last_page = get_last_page(database, ranking_type)
    length = len(str(last_page))

    print('Scraped at:', datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S"))
    for page in tqdm.trange(last_page+1):
        scrape_ranking_page(database, ranking_type, page, fields, tmp_directory, length)
        time.sleep(1)
    merge_anime(tmp_directory, save_file_path)
    
    
def get_last_page(database, ranking_type):

    if database=='anime' and ranking_type=='favorite':
        number_entries =  27_162

    if database=='manga' and ranking_type=='bypopularity':
        number_entries = 59_950

    if database=='manga' and ranking_type=='favorite':
        number_entries = 67_338

    last_page = math.ceil(number_entries / 500) - 1

    params = {'ranking_type': ranking_type, 'limit': 500, 'offset': last_page*500}
    try:
        data = get_data(f'/{database}/ranking', params)
        # ensure data['data'] is not empty and 'next' not in data['paging']
        assert len(data['data']) > 0
        assert 'next' not in data['paging']
    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle the exception, either by returning a default value or handling the error differently.
        data = []  # default value
    
    return last_page

def manga_crash(endpoint, params):
    page = params["offset"]//params["limit"]
    print(f'\n Crashed at page {page}')
    
    params['fields'] = params['fields'].replace('alternative_titles,', '')
    data = get_data(endpoint, params)

    ids = [manga['node']['id'] for manga in data['data']]

    present_fails = [id for id in ids if id in known_fails]

    if not present_fails:
        print('Fail unknown...')
        return data

    offset = page * params['limit']
    problems = [offset-1]
    for fail in known_fails:
        if fail in ids:
            problems.append(offset + ids.index(fail))
    problems.append(offset + params['limit'])

    alternative_titles = []
    params['fields'] = 'alternative_titles'
    for i in range(len(problems)-1):
        params['offset'] = problems[i] + 1
        params['limit'] = problems[i+1] - problems[i] - 1
        data_short = get_data(endpoint, params)
        alternative_titles.extend((manga['node']['id'], manga['node']['alternative_titles']) for manga in data_short['data'])
        time.sleep(1)

    for id, alt_tit in alternative_titles:
        data['data'][ids.index(id)]['node']['alternative_titles'] = alt_tit
    
    return data
