import requests

# # The base directory.
# base_directory = 'src'

# The API URL.
api_url = 'https://api.myanimelist.net/v2'

# Load client ID.
with open(f'client_id.txt', 'r') as f:
    CLIENT_ID = f.read()

headers = {'X-MAL-CLIENT-ID': CLIENT_ID}

def get_data(endpoint, params=None):
    url = api_url + endpoint
    if params:
        url += '?' + '&'.join(f'{key}={value}' for key, value in params.items())
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()
