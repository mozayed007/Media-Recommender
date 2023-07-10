import datetime
import tqdm
import time
import math
import os

from common_utils import keys, known_fails, merge_anime, base_directory
from client import get_data

def scrape_ranking_page(database, ranking_type, page, fields, save_directory, length):
    params = {'ranking_type': ranking_type, 'limit': 500, 'offset': page*500, 'fields': fields}
    try:
        data = get_data(f'/{database}/ranking', params)
    except:
        data = manga_crash(f'/{database}/ranking', params)

    useful = [anime['node'] for anime in data['data']]
    with open(save_directory + f'page{str(page).zfill(length)}.json', 'w') as f:
        json.dump(useful, f)

def scrape_ranking(database='anime', ranking_type='favorite'):

    save_file_path = f'{base_directory}/{database}_mal.json'
    tmp_directory = f'{base_directory}/tmp_{database}_mal'
    os.makedirs(tmp_directory, exist_ok=True)

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
        number_entries =  24_162

    if database=='manga' and ranking_type=='bypopularity':
        number_entries = 59_950

    if database=='manga' and ranking_type=='favorite':
        number_entries = 67_338

    last_page = math.ceil(number_entries / 500) - 1

    params = {'ranking_type': ranking_type, 'limit': 500, 'offset': last_page*500}
    data = get_data(f'/{database}/ranking', params)
    
    assert len(data['data']) > 0
    assert 'next' not in data['paging']

    return last_page

def manga_crash(endpoint, params):
    page = params["offset"]//params["limit"]
    print(f'Crashed at page {page}')
    
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
