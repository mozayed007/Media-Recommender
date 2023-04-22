### Pytorch implementation:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import sentence_transformers
from torch.utils.data import Dataset, DataLoader
import time
from IPython.display import display
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a function to get the text encoding using the GPU
@torch.no_grad()
def get_encoding(x):
    if isinstance(x, str) and x.strip() != "":
        print(f"Encoding: {x}", flush=True)
        sys.stdout.flush()
        encoding = torch.tensor(model.encode([x], show_progress_bar=True), device=device)
        print(f"Encoded tensor shape: {encoding.shape}", flush=True)
        sys.stdout.flush()
        return encoding
    else:
        print(f"Encoding: None", flush=True)
        sys.stdout.flush()
        return None
class MyDataset(Dataset):
    def __init__(self, df, device):
        self.df = df
        self.device = device
        self.df['cleaned_syn_encoding'] = self.df['cleaned_syn'].apply(get_encoding)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['cleaned_syn_encoding'].iloc[idx]
    
# Define a function to get the top k most similar IDs using PyTorch
def get_top_k_similar_IDs(x, df, k=15):
    encoding = x
    if encoding is not None:
        print("Computing cosine similarity distances...", flush=True)
        sys.stdout.flush()
        distances = F.cosine_similarity(encoding, torch.stack(df['cleaned_syn_encoding'].values))
        print(f"Distances tensor shape: {distances.shape}", flush=True)
        sys.stdout.flush()
        closest_paragraphs_indices = distances.argsort(descending=True)[:k + 1][1:]
        print(f"Top {k} indices: {closest_paragraphs_indices}", flush=True)
        sys.stdout.flush()
        return df.iloc[closest_paragraphs_indices]['ID'].values
    else:
        print("Encoding is None, returning None.", flush=True)
        sys.stdout.flush()
        return None



# Define the function to process each chunk
def process_chunk(chunk):
    print(f"Processing chunk {chunk.index[0]}-{chunk.index[-1]}...", flush=True)
    sys.stdout.flush()
    dataset = MyDataset(chunk, device)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
    result = []
    for batch in dataloader:
        with torch.cuda.amp.autocast():
            encodings = torch.stack(batch)
            print(f"Memory allocated: {torch.cuda.memory_allocated(device=device)/(1024**2):.2f} MB", flush=True)  # <-- add this line to check GPU memory usage
            sys.stdout.flush()
            closest_IDs = [get_top_k_similar_IDs(encodings[i], chunk, k=15) for i in range(len(batch))]
        result.extend(closest_IDs)
    chunk['similar_IDs'] = result
    print(f"Memory allocated: {torch.cuda.memory_allocated(device=device)/(1024**2):.2f} MB", flush=True)  # <-- add this line to check GPU memory usage
    sys.stdout.flush()
    return chunk
def init_child(model_, device_):
    print("Initializing child process with model and device...", flush=True)
    sys.stdout.flush()
    global model, device
    model = model_
    device = device_
    print("Initialized child process with model and device.", flush=True)
    sys.stdout.flush()    

if __name__ == '__main__':
    # Set the GPU as the device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    # Load the UniversalSentenceEncoder model
    print("Loading model...")
    model = sentence_transformers.SentenceTransformer('roberta-base')
    model = model.to(device)
    print("Loaded model and moved it to device.")
    print(f"Memory allocated: {torch.cuda.memory_allocated(device=device)/(1024**2):.2f} MB")  # <-- add this line to check GPU memory usage
    # Load the string data:
    df_string = pd.read_csv(r'M:\Anime Recommender\data-history\up-to-date-MAL\anime_string_latest.csv')
    df_string.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
    df_string.dropna(inplace=True)
    # Define the number of processes to use
    num_processes = torch.multiprocessing.cpu_count()
    print(f"Using {num_processes} processes.")
    # Split the data into chunks
    chunks = np.array_split(df_string, num_processes)
    print(f"Data split into {len(chunks)} chunks.")
    # Set up the multiprocessing pool
    print("Initializing multiprocessing pool...")
    pool = torch.multiprocessing.Pool(processes=num_processes, initializer=init_child, initargs=(model, device))
    print("Multiprocessing pool created.", flush=True)
    # Process the chunks in parallel
    results = []
    start_time = time.time()
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...", flush=True)
        sys.stdout.flush()
        result = process_chunk(chunk)
        results.append(result)
        elapsed_time = time.time() - start_time
        print(f"Chunk {i+1}/{len(chunks)} processed in {elapsed_time:.2f} seconds.", flush=True)
        print(f"Memory allocated: {torch.cuda.memory_allocated(device=device)/(1024**2):.2f} MB", flush=True)
        sys.stdout.flush()
        
    # Combine the results from all processes
    final_result = pd.concat(results)
    print("Results combined.")
    # Close the pool
    pool.close()
    pool.join()
    print("Pool closed.")
    # Save the final result to a CSV file
    final_result.to_csv('similar_animes.csv', index=False)
    print("Result saved to file.")