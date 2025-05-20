import argparse
import asyncio
import os
import sys
import pandas as pd
import json
from typing import List, Dict, Any, Optional, Union

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import OptimizedMediaRecommendationEngine
from embedding_model import SentenceTransformerEmbeddingModel
from vector_database import MilvusVectorDatabase
# Assuming configuration loading might be needed later
# from config_loader import load_config # Example, adjust as needed

# Import configuration from centralized config module
from src.config import (
    ANIME_DATA_FILE as DATA_PATH,
    PROCESSED_PICKLE_FILE as PROCESSED_DATA_PICKLE,
    EMBEDDING_MODEL,
    VECTOR_DB,
    COLUMN_NAMES,
    CONTENT_FEATURES,
    get_db_uri
)

# Use values from config
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL['name']
EMBEDDING_DIMENSION = EMBEDDING_MODEL['dimension']
MILVUS_ALIAS = 'default'  # Keep for potential future use
MILVUS_HOST = VECTOR_DB['host']
MILVUS_PORT = VECTOR_DB['port']
MILVUS_URI = get_db_uri()
MILVUS_COLLECTION_NAME = VECTOR_DB['collection_name']
ID_COL = COLUMN_NAMES['id']
TITLE_COL = COLUMN_NAMES['title']  # 'title' in the actual data
DESC_COL = COLUMN_NAMES['description']  # 'synopsis' in the actual data
# --- End Configuration ---


def parse_list_arg(arg_string: str) -> List[str]:
    """Parse a comma-separated string into a list of strings."""
    if not arg_string:
        return []
    return [item.strip() for item in arg_string.split(',') if item.strip()]

def parse_range_arg(arg_string: str) -> tuple:
    """Parse a range string like '7.5-10' into a tuple of (min, max)."""
    if not arg_string or '-' not in arg_string:
        return None
    try:
        parts = arg_string.split('-')
        if len(parts) != 2:
            return None
        min_val = float(parts[0].strip())
        max_val = float(parts[1].strip())
        return (min_val, max_val)
    except ValueError:
        return None

def setup_recommendation_args_and_engine(args):
    """Processes CLI arguments, creates filters, and initializes the engine."""
    filters = {}
    
    # --- Process filter arguments --- 
    # Process list filters
    for list_filter in ['genres', 'themes', 'demographics', 'studios']:
        arg_value = getattr(args, list_filter, None)
        if arg_value:
            filters[list_filter] = parse_list_arg(arg_value)
    
    # Process range filters
    for range_filter, column_name in [
        ('score-range', 'score'),
        ('year-range', 'start_year'),
        ('episodes-range', 'episodes')
    ]:
        arg_value = getattr(args, range_filter.replace('-', '_'), None)
        if arg_value:
            parsed_range = parse_range_arg(arg_value)
            if parsed_range:
                filters[column_name + '_min'] = parsed_range[0]
                filters[column_name + '_max'] = parsed_range[1]
    
    # Process single value filters
    if args.media_type:
        media_type = args.media_type.lower()
        filters['media_type'] = media_type
        
    for single_filter in ['status', 'rating']:
        arg_value = getattr(args, single_filter, None)
        if arg_value:
            filters[single_filter] = arg_value
    
    # Process boolean filters
    if args.sfw_only:
        filters['sfw'] = True

    # Process JSON filter if provided
    if args.filter_json:
        try:
            import json
            json_filters = json.loads(args.filter_json)
            # Merge JSON filters, potentially overriding CLI filters
            filters.update(json_filters)
            print(f"Applied filters from JSON: {json_filters}")
        except json.JSONDecodeError as e:
            print(f"Error parsing --filter-json: {e}. Ignoring JSON filter.")
        except Exception as e:
            print(f"Error processing --filter-json: {e}. Ignoring JSON filter.")

    # --- Initialize Components (can be mocked in tests) ---
    print("Initializing components...")
    embedding_model = SentenceTransformerEmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
    # Correct the instantiation of MilvusVectorDatabase
    vector_db = MilvusVectorDatabase(
        embedding_model=embedding_model,
        uri=MILVUS_URI, # Pass the constructed URI
        collection_name=MILVUS_COLLECTION_NAME,
        dimension=EMBEDDING_DIMENSION # Pass the embedding dimension
    )
    engine = OptimizedMediaRecommendationEngine(
        embedding_model=embedding_model,
        vector_db=vector_db,
        data_path=DATA_PATH,
        processed_data_path=PROCESSED_DATA_PICKLE,
        id_col=ID_COL,
        title_col=TITLE_COL,
        desc_col=DESC_COL,
        content_feature_cols=CONTENT_FEATURES
    )
    print("Components initialized.")
    return filters, engine

async def main():
    parser = argparse.ArgumentParser(description="Media Recommender CLI")
    parser.add_argument("--type", type=str, required=True, choices=['semantic-desc', 'semantic-title', 'content', 'combined'], 
                        help="Type of recommendation to generate.")
    parser.add_argument("--query", type=str, help="Description query for semantic search.")
    parser.add_argument("--title", type=str, help="Title for semantic search.")
    parser.add_argument("--id", type=int, help="anime_id for content-based or combined search.")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor for combined recommendations (0=content, 1=semantic).")
    
    # Filtering options
    filter_group = parser.add_argument_group('Filtering options')
    filter_group.add_argument("--genres", type=str, help="Filter by genres (comma-separated list)")
    filter_group.add_argument("--themes", type=str, help="Filter by themes (comma-separated list)")
    filter_group.add_argument("--demographics", type=str, help="Filter by demographics (comma-separated list)")
    filter_group.add_argument("--studios", type=str, help="Filter by studios (comma-separated list)")
    filter_group.add_argument("--score-range", type=str, help="Filter by score range (format: min-max, e.g. '7.5-10')")
    filter_group.add_argument("--year-range", type=str, help="Filter by year range (format: min-max, e.g. '2010-2023')")
    filter_group.add_argument("--media-type", type=str, help="Filter by media type (TV, Movie, OVA, etc.)")
    filter_group.add_argument("--status", type=str, help="Filter by status (Airing, Finished, etc.)")
    filter_group.add_argument("--episodes-range", type=str, help="Filter by episode count range (format: min-max, e.g. '12-24')")
    filter_group.add_argument("--rating", type=str, help="Filter by content rating (PG, PG-13, R, etc.)")
    filter_group.add_argument("--sfw-only", action="store_true", help="Only include Safe-For-Work content")
    filter_group.add_argument("--filter-json", type=str, help="Advanced filtering with JSON format")

    args = parser.parse_args()

    # --- Validation ---
    if args.type in ['semantic-desc', 'combined'] and not args.query and not args.title and not args.id:
        parser.error("--query or --title is required for semantic-desc/combined if --id is not provided.")
    if args.type == 'semantic-title' and not args.title:
        parser.error("--title is required for semantic-title.")
    if args.type in ['content', 'combined'] and args.id is None:
        parser.error("--id is required for content/combined recommendations.")
    if args.type == 'combined' and not (args.id and (args.query or args.title)):
        print("Warning: Running combined without both content (--id) and semantic (--query/--title) inputs.")
    
    # --- Process filter arguments and Initialize Components using helper ---
    try:
        filters, engine = setup_recommendation_args_and_engine(args)
    except Exception as setup_error:
        print(f"Error during setup: {setup_error}")
        sys.exit(1)

    # --- Load Data ---
    print("Loading data into recommendation engine...")
    try:
        await engine.load_data()
        print("Data loading complete.")
        # Verify content filter is loaded if needed
        if args.type in ['content', 'combined'] and engine.content_filter is None:
            print("Error: Content filter failed to initialize during data loading.")
            return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Get Recommendations ---
    recommendations = []
    print(f"\nFetching {args.k} recommendations of type '{args.type}'...")
    try:
        # Get initial recommendations based on type
        if args.type == 'semantic-desc':
            if not args.query:
                print("Error: --query is required for semantic-desc.")
                return
            recommendations = await engine.get_recommendations_by_description(args.query, k=args.k * 3)  # Get more for filtering
        elif args.type == 'semantic-title':
            if not args.title:
                print("Error: --title is required for semantic-title.")
                return
            recommendations = await engine.get_recommendations_by_title(args.title, k=args.k * 3)  # Get more for filtering
        elif args.type == 'content':
            if args.id is None:
                print("Error: --id is required for content.")
                return
            recommendations = engine.get_content_based_recommendations(args.id, k=args.k * 3)  # Get more for filtering
        elif args.type == 'combined':
            # Ensure at least one input type is present for combined
            if args.id is None and not args.query and not args.title:
                print("Error: --id or --query or --title required for combined.")
                return
            recommendations = await engine.get_combined_recommendations(
                item_id=args.id,
                query=args.query,
                title=args.title,
                k=args.k * 3,  # Get more for filtering
                alpha=args.alpha
            )
        
        # Apply filters if any are specified
        if filters and recommendations:
            print("Applying filters to recommendations...")
            filtered_recommendations = []
            item_ids = [item[0] for item in recommendations]
            
            # Get detailed information for all recommended items
            item_details = []
            for item_id in item_ids:
                details = engine.get_item_details(item_id)
                if details:  # Only include items with valid details
                    item_details.append(details)
            
            # Create DataFrame for easier filtering
            df_details = pd.DataFrame(item_details)
            
            # Apply filters to the DataFrame
            if df_details.empty:
                print("Warning: No item details available for filtering")
            else:
                # Print the actual columns available in the DataFrame for debugging
                print(f"Available columns for filtering: {df_details.columns.tolist()}")
                
                # Apply list filters (genres, themes, demographics, studios)
                # Map field names to exact column names based on data inspection
                # NOTE: All column names are lowercase in the actual data
                field_name_mapping = {
                    'genres': ['genres'],
                    'themes': ['themes'],
                    'demographics': ['demographics'],
                    'studios': ['studios']
                }
                
                for list_field, filter_values in filters.items():
                    if list_field not in field_name_mapping:
                        continue
                        
                    # Try all possible column name variants
                    found = False
                    for column_name in field_name_mapping[list_field]:
                        if column_name in df_details.columns:
                            # Case-insensitive matching of filter values with list elements
                            try:
                                filter_values_lower = [val.lower() for val in filter_values] if isinstance(filter_values, list) else []
                                
                                def check_list_match(x):
                                    """Check if any of the filter values match items in the list x."""
                                    # Handle numpy array explicitly first
                                    if isinstance(x, np.ndarray):
                                        try:
                                            x = x.tolist()
                                        except:
                                            x = [item for item in x]
                                    
                                    # Check for None or scalar NaN *after* potentially converting array
                                    if x is None or (not isinstance(x, (list, tuple)) and pd.isna(x)):
                                        return False
                                    
                                    # If it's a list or converted numpy array
                                    if isinstance(x, (list, tuple)):
                                        # If empty list, no match
                                        if not x:
                                            return False
                                        
                                        # Loop through each filter value and item in the list
                                        for filter_val in filter_values_lower:
                                            for item in x:
                                                # Convert item to string if not already
                                                if not isinstance(item, str):
                                                    item = str(item)
                                                if filter_val.lower() in item.lower():
                                                    return True
                                        return False
                                    
                                    # If it's some other type, convert to string and check
                                    try:
                                        item_str = str(x).lower()
                                        for filter_val in filter_values_lower:
                                            if filter_val.lower() in item_str:
                                                return True
                                    except:
                                        pass
                                        
                                    return False
                                        
                                    return False
                                    
                                # Handle the filter differently based on whether we have numpy arrays or not
                                try:
                                    # Import numpy here to ensure it's in scope
                                    import numpy as np
                                    
                                    # Check if we're dealing with numpy arrays
                                    if len(df_details) > 0 and isinstance(df_details[column_name].iloc[0], np.ndarray):
                                        # For numpy arrays, we need to be extra careful
                                        matches = []
                                        for idx, row_val in df_details[column_name].items():
                                            # Apply the check_list_match function to each row
                                            matches.append(check_list_match(row_val))
                                        # Create a boolean Series for filtering
                                        mask = pd.Series(matches, index=df_details.index)
                                    else:
                                        # For other types, the regular apply should work
                                        mask = df_details[column_name].apply(lambda x: check_list_match(x))
                                except Exception as filter_err:
                                    print(f"Error in filter application: {filter_err}")
                                    # Fallback to a more explicit loop if apply fails
                                    matches = []
                                    for idx, row_val in df_details[column_name].items():
                                        try:
                                            matches.append(check_list_match(row_val))
                                        except:
                                            matches.append(False)  # Assume no match if error
                                    mask = pd.Series(matches, index=df_details.index)
                            except Exception as e:
                                print(f"Error applying filter '{list_field}': {e}")
                                continue
                            df_details = df_details[mask]
                            found = True
                            print(f"Applied filter '{list_field}' using column '{column_name}', {len(df_details)} items remaining")
                            break
                    
                    if not found and list_field in ['genres', 'themes', 'demographics', 'studios']:
                        print(f"Warning: Could not find column for '{list_field}' filter")
                
                # Apply range filters
                # NOTE: All column names are lowercase in the actual data
                field_name_mapping = {
                    'score': ['score'],
                    'start_year': ['start_year'],
                    'episodes': ['episodes']
                }
                
                for field, column_variants in field_name_mapping.items():
                    min_key = f"{field}_min"
                    max_key = f"{field}_max"
                    
                    if min_key in filters or max_key in filters:
                        # Try all possible column variants
                        found = False
                        for column_name in column_variants:
                            if column_name in df_details.columns:
                                if min_key in filters:
                                    df_details = df_details[df_details[column_name] >= filters[min_key]]
                                    print(f"Applied min filter for '{field}' using column '{column_name}', {len(df_details)} items remaining")
                                    
                                if max_key in filters:
                                    df_details = df_details[df_details[column_name] <= filters[max_key]]
                                    print(f"Applied max filter for '{field}' using column '{column_name}', {len(df_details)} items remaining")
                                    
                                found = True
                                break
                                
                        if not found:
                            print(f"Warning: Could not find column for '{field}' filter")
                
                # Apply single value filters (media_type, status, rating, sfw)
                # NOTE: All column names are lowercase in the actual data
                field_name_mapping = {
                    'media_type': ['type'], # Column is 'type' in the data
                    'status': ['status'],
                    'rating': ['rating']
                }
                
                for field, column_variants in field_name_mapping.items():
                    if field in filters:
                        # Try all possible column variants
                        found = False
                        for column_name in column_variants:
                            if column_name in df_details.columns:
                                # Case-insensitive matching for string fields
                                try:
                                    if df_details[column_name].dtype == 'object':
                                        # Handle potential NaN values
                                        mask = df_details[column_name].fillna('').astype(str).str.lower() == filters[field].lower()
                                    else:
                                        mask = df_details[column_name] == filters[field]
                                except Exception as e:
                                    print(f"Error applying filter '{field}': {e}")
                                    continue
                                
                                df_details = df_details[mask]
                                print(f"Applied filter '{field}' using column '{column_name}', {len(df_details)} items remaining")
                                found = True
                                break
                                
                        if not found:
                            print(f"Warning: Could not find column for '{field}' filter")
                
                # Apply boolean filters
                if 'sfw' in filters:
                    # Try different column names for SFW status
                    sfw_columns = ['sfw', 'is_sfw', 'nsfw', 'is_nsfw']
                    found = False
                    
                    for column_name in sfw_columns:
                        if column_name in df_details.columns:
                            # If column is 'nsfw' or 'is_nsfw', we need to invert the logic
                            try:
                                if column_name.lower().find('nsfw') >= 0:
                                    df_details = df_details[df_details[column_name].fillna(False) != filters['sfw']]
                                else:
                                    df_details = df_details[df_details[column_name].fillna(True) == filters['sfw']]
                            except Exception as e:
                                print(f"Error applying SFW filter: {e}")
                                continue
                            print(f"Applied SFW filter using column '{column_name}', {len(df_details)} items remaining")
                            found = True
                            break
                    
                    if not found:
                        print("Warning: Could not find column for SFW filter")
                
                # Get filtered IDs
                if df_details.empty:
                    print("Warning: After applying filters, no items match the criteria")
                    print("Trying with relaxed filter criteria...")
                    
                    # Get a fresh copy of details
                    item_details = []
                    for item_id in item_ids:
                        details = engine.get_item_details(item_id)
                        if details:  # Only include items with valid details
                            item_details.append(details)
                    
                    # Try again with just primary filters
                    df_details = pd.DataFrame(item_details)
                    
                    # Start with SFW filter if specified (this is usually a hard requirement)
                    if 'sfw' in filters and filters['sfw'] is True:
                        # Apply SFW filter
                        sfw_columns = ['sfw', 'is_sfw', 'safe_for_work']
                        found = False
                        for column_name in sfw_columns:
                            if column_name in df_details.columns:
                                # If column is 'nsfw' or 'is_nsfw', invert the logic
                                try:
                                    if column_name.lower().find('nsfw') >= 0:
                                        df_details = df_details[df_details[column_name].fillna(False) != filters['sfw']]
                                    else:
                                        df_details = df_details[df_details[column_name].fillna(True) == filters['sfw']]
                                except Exception as e:
                                    print(f"Error applying SFW filter: {e}")
                                    continue
                                print(f"Applied SFW filter using column '{column_name}', {len(df_details)} items remaining")
                                found = True
                                break
                    
                    # Then try with genre filters only if specified
                    if 'genres' in filters and len(df_details) > 0:
                        column_name = 'genres'  # We know this exists from earlier checks
                        try:
                            import numpy as np
                            filter_values = filters['genres']
                            filter_values_lower = [val.lower() for val in filter_values] if isinstance(filter_values, list) else []
                            
                            # Define a safer check function for relaxed filtering
                            def safe_check_match(row_val):
                                """Check if any of the filter values match items in the list x."""
                                # Import numpy inside function for proper scope
                                import numpy as np
                                
                                try:
                                    # Convert numpy arrays to lists
                                    if isinstance(row_val, np.ndarray):
                                        row_list = row_val.tolist()
                                    elif isinstance(row_val, list):
                                        row_list = row_val
                                    else:
                                        # String or other type - just use as is in check_list_match
                                        return check_list_match(row_val)
                                    
                                    # Manual string comparison to avoid ambiguous truth value errors
                                    for item in row_list:
                                        if isinstance(item, str):
                                            item_lower = item.lower()
                                            for genre in filter_values_lower:
                                                if genre.lower() in item_lower:
                                                    return True
                                    return False
                                except Exception as e:
                                    print(f"  - Error in safe_check_match: {e}")
                                    return False  # Safety default
                            
                            # Apply the safer function
                            matches = []
                            for idx, row_val in df_details[column_name].items():
                                matches.append(safe_check_match(row_val))
                            mask = pd.Series(matches, index=df_details.index)
                            df_details = df_details[mask]
                            print(f"Applied relaxed filter 'genres' using column '{column_name}', {len(df_details)} items remaining")
                        except Exception as e:
                            print(f"Error applying relaxed genre filter: {e}")
                    
                    # Check results after relaxed filtering
                    if df_details.empty:
                        print("Warning: Even with relaxed criteria, no items match the filters")
                        filtered_ids = set()
                    else:
                        filtered_ids = set(df_details[engine.id_col].tolist())
                else:
                    filtered_ids = set(df_details[engine.id_col].tolist())
                
                # Update recommendations list with only the filtered items, preserving order and scores
                filtered_recommendations = [rec for rec in recommendations if rec[0] in filtered_ids]
                
                if not filtered_recommendations:
                    print("Warning: No recommendations match the specified filters")
                    # Still return some recommendations even if they don't match all filters
                    print("Returning results without filtering...")
                    recommendations = recommendations[:args.k]
                    return
                
                recommendations = filtered_recommendations[:args.k]  # Limit to requested k
                print(f"After filtering: {len(recommendations)} recommendations match criteria")
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return

    # --- Display Results ---
    print("\n--- Recommendations ---")
    if recommendations:
        # Sort recommendations by score (assuming the score is a distance metric where lower is better)
        # But display similarity where higher is better (1.0 - normalized_distance)
        if args.type in ['semantic-desc', 'semantic-title']:
            # For semantic search, sort by distance (lower is better)
            sorted_recommendations = sorted(recommendations, key=lambda x: x[2])
        
        # Create a DataFrame for better display
        df_recs = pd.DataFrame(sorted_recommendations[:args.k], columns=['ID', 'Title', 'Score'])
        # For semantic search, convert distance to similarity score
        if args.type in ['semantic-desc', 'semantic-title']:
            # Normalize to similarity score (0-1 where 1 is most similar)
            # Avoid division by zero
            if len(df_recs) > 0:
                max_score = df_recs['Score'].max()
                if max_score > 0:
                    df_recs['Score'] = 1.0 - (df_recs['Score'] / max_score)
                else:
                    # If all scores are 0, set to high similarity
                    df_recs['Score'] = 1.0
        
        df_recs['Score'] = df_recs['Score'].round(4)
        print(df_recs.to_string(index=False))
    else:
        print("No recommendations found.")

    # --- Cleanup (Optional) ---
    # Consider disconnecting Milvus if necessary, though often managed by context/lifetime
    # await vector_db.disconnect()
    print("\nCLI finished.")


if __name__ == "__main__":
    # Check if Milvus/dependencies are running (basic check)
    # Add more robust checks if needed
    print("Ensure Milvus service is running.")
    asyncio.run(main())
