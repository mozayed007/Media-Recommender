import json
import os

# The base directory.
base_directory = '/data/raw'

# The dictionary containing the keys needed.
common_keys = [
    'id', 'title', 'main_picture', 'alternative_titles', 'start_date', 'end_date', 'synopsis', 'mean',
    'rank', 'popularity', 'num_list_users', 'num_scoring_users', 'num_favorites', 'nsfw', 'genres', 
    'created_at', 'updated_at', 'media_type', 'status'
]

anime_keys = [*common_keys, 'num_episodes', 'start_season', 'broadcast', 'source', 
                'average_episode_duration', 'rating', 'studios']

manga_keys = [*common_keys, 'num_volumes', 'num_chapters', 'authors{id,first_name,last_name}']

keys = {'anime': anime_keys, 'manga': manga_keys}

# The known failures.
known_fails = [116770, 144472, 115838, 143751, 146583, 148716]

def merge_anime(tmp_directory, save_file_path):
    data = []
    
    for file_name in os.listdir(tmp_directory):
        file_path = os.path.join(tmp_directory, file_name)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
        data.extend(file_data)

    with open(save_file_path, 'w') as f:
        json.dump(data, f, indent=4)
