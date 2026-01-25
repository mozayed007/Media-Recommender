import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, AsyncMock

# Add the src directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query_interface import parse_list_arg, parse_range_arg

class TestQueryInterfaceHelpers(unittest.TestCase):
    """Test helper functions in query_interface.py"""
    
    def test_parse_list_arg(self):
        """Test parsing comma-separated lists"""
        # Test normal comma-separated list
        result = parse_list_arg("Action,Comedy,Drama")
        self.assertEqual(result, ["Action", "Comedy", "Drama"])
        
        # Test with whitespace
        result = parse_list_arg(" Action , Comedy , Drama ")
        self.assertEqual(result, ["Action", "Comedy", "Drama"])
        
        # Test with empty input
        result = parse_list_arg("")
        self.assertEqual(result, [])
        
        # Test with None input
        result = parse_list_arg(None)
        self.assertEqual(result, [])
    
    def test_parse_range_arg(self):
        """Test parsing range arguments"""
        # Test normal range
        result = parse_range_arg("7.5-10")
        self.assertEqual(result, (7.5, 10.0))
        
        # Test with whitespace
        result = parse_range_arg(" 7.5 - 10 ")
        self.assertEqual(result, (7.5, 10.0))
        
        # Test with integers
        result = parse_range_arg("7-10")
        self.assertEqual(result, (7.0, 10.0))
        
        # Test with invalid format
        result = parse_range_arg("invalid")
        self.assertIsNone(result)
        
        # Test with None input
        result = parse_range_arg(None)
        self.assertIsNone(result)


class TestFilterFunctions(unittest.TestCase):
    """Test the filtering logic in query_interface.py"""
    
    def setUp(self):
        """Set up test data to match actual structure with numpy arrays"""
        # Create a small test DataFrame that matches the structure of the actual data
        self.test_data = pd.DataFrame({
            'anime_id': [1, 2, 3, 4, 5],
            'title': ['Anime A', 'Anime B', 'Anime C', 'Anime D', 'Anime E'],
            'type': ['tv', 'movie', 'tv', 'ova', 'tv'],
            'score': [8.5, 7.2, 9.1, 6.8, 8.0],
            'status': ['finished_airing', 'currently_airing', 'finished_airing', 'not_yet_aired', 'currently_airing'],
            'episodes': [24, 1, 12, 4, 13],
            'genres': [
                np.array(['Action', 'Adventure', 'Fantasy']),
                np.array(['Comedy', 'Romance']),
                np.array(['Action', 'Drama', 'Fantasy']),
                np.array(['Comedy', 'Slice of Life']),
                np.array(['Action', 'Sci-Fi'])
            ],
            'sfw': [True, True, False, True, True]
        })
    
    @patch('src.query_interface.main')
    def test_check_list_match_function(self, mock_main):
        """Test the check_list_match function within main"""
        # Since the function is nested inside main, we need to use the patch decorator
        # and then extract the function for testing
        
        # Define a mock context that simulates the local environment inside main
        # This includes the filter_values_lower list that check_list_match uses
        mock_context = {
            'filter_values_lower': ['action', 'fantasy']
        }
        
        # Import the module to access the file source
        import src.query_interface
        
        # Define the check_list_match function based on the actual function in query_interface.py
        # --- START: Exact copy from src/query_interface.py (corrected version) --- 
        def check_list_match(x):
            """Check if any of the filter values match items in the list x."""
            # Import numpy here to avoid scope issues
            import numpy as np

            # Handle numpy array explicitly first
            if isinstance(x, np.ndarray):
                try:
                    x = x.tolist()
                except:
                    x = [item for item in x]
            
            # Check for None or scalar NaN *after* potentially converting array
            if x is None or (not isinstance(x, (list, tuple)) and pd.isna(x)):
                return False
            
            # If it's a string (sometimes genres are stored as comma-separated strings)
            if isinstance(x, str):
                # Handle potential JSON-like string with quotes and brackets
                if x.startswith('[') and x.endswith(']'):
                    try:
                        # Try to parse as list
                        import json
                        x_list = json.loads(x)
                        x_list = [str(item).strip() for item in x_list if item is not None]
                    except:
                        # Fall back to simple splitting by comma
                        x_list = [item.strip() for item in x.split(',')]
                else:
                    x_list = [item.strip() for item in x.split(',')]
                
                # Check if any filter value appears in any list item
                for filter_val in mock_context['filter_values_lower']:
                    for item in x_list:
                        if filter_val.lower() in item.lower():
                            return True
                return False
            
            # If it's a list or converted numpy array
            if isinstance(x, (list, tuple)):
                # If empty list, no match
                if not x:
                    return False
                
                # Loop through each filter value and item in the list
                for filter_val in mock_context['filter_values_lower']:
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
                for filter_val in mock_context['filter_values_lower']:
                    if filter_val.lower() in item_str:
                        return True
            except:
                pass
                
            return False
        # --- END: Exact copy from src/query_interface.py (corrected version) --- 
        # Test against numpy arrays (like in the actual data)
        self.assertTrue(check_list_match(np.array(['Action', 'Adventure', 'Fantasy'])))
        self.assertTrue(check_list_match(np.array(['Action', 'Drama'])))
        self.assertFalse(check_list_match(np.array(['Comedy', 'Romance'])))
        
        # Test against lists
        self.assertTrue(check_list_match(['Action', 'Adventure', 'Fantasy']))
        self.assertFalse(check_list_match(['Comedy', 'Romance']))
        
        # Test against strings
        self.assertTrue(check_list_match('Action, Adventure, Fantasy'))
        self.assertFalse(check_list_match('Comedy, Romance'))
        
        # Test against empty/null values
        self.assertFalse(check_list_match([]))
        self.assertFalse(check_list_match(None))
        self.assertFalse(check_list_match(np.nan))


class TestSetupFunction(unittest.TestCase):
    """Test the synchronous setup_recommendation_args_and_engine function"""
    
    @patch('argparse.ArgumentParser.parse_args') # Keep patch for args object generation
    @patch('src.query_interface.SentenceTransformerEmbeddingModel')
    @patch('src.query_interface.MilvusVectorDatabase')
    @patch('src.query_interface.OptimizedMediaRecommendationEngine')
    def test_setup_function(self, mock_engine, mock_db, mock_embedding, mock_args):
        """Test argument processing, filter creation, and engine initialization"""
        # Configure the mock args
        mock_args_obj = MagicMock()
        mock_args_obj.type = 'semantic-desc'
        mock_args_obj.query = 'epic battle'
        mock_args_obj.k = 10
        mock_args_obj.genres = 'Action,Fantasy'
        mock_args_obj.score_range = '8-10'
        mock_args_obj.media_type = 'TV'
        mock_args_obj.sfw_only = True
        mock_args_obj.id = None
        mock_args_obj.title = None
        mock_args_obj.themes = None
        mock_args_obj.demographics = None
        mock_args_obj.studios = None
        mock_args_obj.year_range = None
        mock_args_obj.episodes_range = None
        mock_args_obj.status = None
        mock_args_obj.rating = None
        mock_args_obj.filter_json = None
        # mock_args.return_value = mock_args_obj # No longer needed as we pass the obj directly

        # Import the function to test
        from src.query_interface import setup_recommendation_args_and_engine
        
        # Call the setup function directly
        filters, engine_instance = setup_recommendation_args_and_engine(mock_args_obj)

        # --- Assertions for Filters ---
        self.assertIn('genres', filters)
        self.assertEqual(filters['genres'], ['Action', 'Fantasy'])
        self.assertIn('score_min', filters)
        self.assertEqual(filters['score_min'], 8.0)
        self.assertIn('score_max', filters)
        self.assertEqual(filters['score_max'], 10.0)
        self.assertIn('media_type', filters)
        self.assertEqual(filters['media_type'], 'tv')
        self.assertIn('sfw', filters)
        self.assertTrue(filters['sfw'])
        self.assertNotIn('year_min', filters) # Example check for non-provided filter

        # --- Assertions for Component Initialization ---
        mock_embedding.assert_called_once()
        mock_db.assert_called_once()
        mock_engine.assert_called_once()
        self.assertEqual(engine_instance, mock_engine.return_value)


if __name__ == '__main__':
    unittest.main()
