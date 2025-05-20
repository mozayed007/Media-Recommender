import pandas as pd
import numpy as np
import json
import os

# Paths to data files
PARQUET_PATH = r"e:\Repos\Media-Recommender\data\raw\anime_Aug24.parquet"
CSV_PATH = r"e:\Repos\Media-Recommender\data\raw\anime_Aug24.csv"

def inspect_parquet_file():
    """Examine the structure of the parquet file in detail."""
    print(f"\n=== Inspecting Parquet File: {PARQUET_PATH} ===")
    try:
        df = pd.read_parquet(PARQUET_PATH)
        print(f"Shape: {df.shape}")
        
        # Print column names and data types
        print("\nColumns and Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Show the first 3 rows for a detailed view
        print("\nFirst 3 rows (truncated):")
        sample_df = df.head(3)
        
        # Display in a readable format
        for idx, row in sample_df.iterrows():
            print(f"\n--- Row {idx} ---")
            for col in df.columns:
                val = row[col]
                
                # Format special types for better readability
                if isinstance(val, np.ndarray):
                    val = f"np.array({val.tolist()})"
                elif isinstance(val, list):
                    val = f"list{val}"
                elif pd.isna(val):
                    val = "NULL"
                elif isinstance(val, str) and len(val) > 100:
                    val = val[:100] + "..."
                
                print(f"  {col}: {val}")
        
        # Show special column analysis
        print("\n=== Special Column Analysis ===")
        if 'genres' in df.columns:
            print("\nGenres Analysis:")
            genre_types = df['genres'].apply(type).value_counts()
            print(f"Data types in 'genres' column: {genre_types}")
            
            # Show some example values
            print("\nExample genre values:")
            for i, genres in enumerate(df['genres'].head(5)):
                print(f"  Row {i}: {type(genres).__name__} - {genres}")
        
        if 'score' in df.columns:
            print("\nScore Analysis:")
            score_stats = df['score'].describe()
            print(score_stats)
            
        if 'type' in df.columns:
            print("\nType Analysis (media type):")
            type_counts = df['type'].value_counts()
            print(type_counts.head(10))
    
    except Exception as e:
        print(f"Error inspecting parquet file: {e}")

def inspect_csv_file():
    """Examine the structure of the CSV file."""
    print(f"\n=== Inspecting CSV File: {CSV_PATH} ===")
    try:
        df = pd.read_csv(CSV_PATH, nrows=3)  # Read just 3 rows for quick inspection
        print(f"Shape (first 3 rows): {df.shape}")
        
        # Print column names
        print("\nColumns:")
        print(df.columns.tolist())
        
        # Show the first 3 rows
        print("\nFirst 3 rows preview:")
        print(df.head(3).to_string())
        
    except Exception as e:
        print(f"Error inspecting CSV file: {e}")

if __name__ == "__main__":
    inspect_parquet_file()
    inspect_csv_file()
