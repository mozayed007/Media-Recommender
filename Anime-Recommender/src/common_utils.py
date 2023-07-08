import requests
import json
import os

api_url = 'https://api.myanimelist.net/v2'

# A Client ID is needed (https://myanimelist.net/apiconfig)
with open('client_id.txt', 'r') as f:
    CLIENT_ID = f.read()

headers = {'X-MAL-CLIENT-ID': CLIENT_ID}

def get_data(endpoint, params=None):
    url = api_url + endpoint
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def merge_files(tmp_directory, save_file_path):

    data = []
    for file_name in os.listdir(tmp_directory):
        file_path = os.path.join(tmp_directory, file_name)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
        data.extend(file_data)

    with open(save_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    #shutil.rmtree(tmp_directory) removed shutil but added it again
