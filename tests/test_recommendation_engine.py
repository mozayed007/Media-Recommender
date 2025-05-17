import os
import sys
import unittest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
import pandas as pd
import numpy as np
import asyncio

# Add the src directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestOptimizedMediaRecommendationEngine(unittest.IsolatedAsyncioTestCase):
    """Test the OptimizedMediaRecommendationEngine class"""
    
    async def asyncSetUp(self):
        """Set up mocks and test data"""
        # Create mock components
        self.mock_embedding_model = MagicMock()
        self.mock_vector_db = MagicMock()
        self.mock_vector_db.add_items = AsyncMock()
        self.mock_vector_db.find_similar_by_description = AsyncMock()
        self.mock_vector_db.count = AsyncMock(return_value=0) # Configure count as AsyncMock, returning 0 initially
        
        # Set up test data paths
        self.test_data_path = "test_data.parquet"
        self.test_processed_path = "test_processed.pkl"
        
        # Create a test DataFrame
        self.test_df = pd.DataFrame({
            'anime_id': [1, 2, 3, 4, 5],
            'title': ['Anime A', 'Anime B', 'Anime C', 'Anime D', 'Anime E'],
            'synopsis': ['Description A', 'Description B', 'Description C', 'Description D', 'Description E'],
            'genres': [
                np.array(['Action', 'Adventure', 'Fantasy']),
                np.array(['Comedy', 'Romance']),
                np.array(['Action', 'Drama', 'Fantasy']),
                np.array(['Comedy', 'Slice of Life']),
                np.array(['Action', 'Sci-Fi'])
            ],
            'studios': [
                np.array(['Studio X']),
                np.array(['Studio Y']),
                np.array(['Studio Z']),
                np.array(['Studio X']),
                np.array(['Studio Y'])
            ],
            'score': [8.5, 7.2, 9.1, 6.8, 8.0],
            'episodes': [24, 1, 12, 4, 13],
            'type': ['tv', 'movie', 'tv', 'ova', 'tv'],
            'status': ['finished_airing', 'currently_airing', 'finished_airing', 'not_yet_aired', 'currently_airing'],
            'sfw': [True, True, False, True, True]
        })
        
        # Create ID to title mapping
        self.id_to_title = {
            1: 'Anime A',
            2: 'Anime B',
            3: 'Anime C',
            4: 'Anime D',
            5: 'Anime E'
        }
        
        # Import the class to test
        from src.recommendation_engine import OptimizedMediaRecommendationEngine
        self.engine_class = OptimizedMediaRecommendationEngine
        
        # Patch the MediaDataset class
        self.mock_dataset = MagicMock()
        self.mock_dataset.get_dataframe.return_value = self.test_df
        
        # Create patcher for the MediaDataset import
        self.dataset_patcher = patch('src.recommendation_engine.MediaDataset')
        self.mock_dataset_class = self.dataset_patcher.start()
        self.mock_dataset_class.return_value = self.mock_dataset
        
        # Create patcher for ContentBasedFilter
        self.cbf_patcher = patch('src.recommendation_engine.ContentBasedFilter')
        self.mock_cbf = self.cbf_patcher.start()
        
        # Create a recommendation engine instance for testing
        self.engine = self.engine_class(
            embedding_model=self.mock_embedding_model,
            vector_db=self.mock_vector_db,
            data_path=self.test_data_path,
            processed_data_path=self.test_processed_path,
            id_col='anime_id',
            title_col='title',
            desc_col='synopsis',
            content_feature_cols={
                'text': ['genres', 'studios'],
                'numeric': ['score', 'episodes']
            }
        )
        self.engine.media_df = self.test_df  # Set mock media_df directly
        
    def tearDown(self):
        """Clean up after tests"""
        # Stop patchers
        self.dataset_patcher.stop()
        self.cbf_patcher.stop()
    
    @patch('os.path.exists')
    @patch('pickle.dump')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    async def test_load_processed_data(self, mock_file, mock_pickle_load, mock_pickle_dump, mock_exists):
        """Test loading processed data from pickle file"""
        # Setup mock to indicate the pickle file exists
        mock_exists.return_value = True
        
        # Set up mock pickle data
        mock_data = [(1, 'Anime A', 'Description A'), (2, 'Anime B', 'Description B')]
        mock_pickle_load.return_value = mock_data
        
        # Test loading data
        result = self.engine.load_processed_data()
        
        # Assertions
        self.assertTrue(result)
        self.assertEqual(self.engine.media_data, mock_data)
        mock_file.assert_called_once_with(self.test_processed_path, 'rb')
        mock_pickle_load.assert_called_once()
        
        # Test when file doesn't exist
        mock_exists.return_value = False
        result = self.engine.load_processed_data()
        self.assertFalse(result)
    
    async def test_load_data(self):
        """Test loading DataFrame and initializing ContentBasedFilter"""
        # Call the method to test
        await self.engine.load_data()
        
        # Assertions
        self.mock_dataset.get_dataframe.assert_called_once()
        self.assertEqual(self.engine.media_df.shape, self.test_df.shape)
        
        # Check ContentBasedFilter initialization
        self.mock_cbf.assert_called_once()
        # Check parameters to ContentBasedFilter
        args, kwargs = self.mock_cbf.call_args
        self.assertEqual(kwargs['id_col'], 'anime_id')
        self.assertEqual(kwargs['text_feature_cols'], ['genres', 'studios'])
        self.assertEqual(kwargs['numeric_feature_cols'], ['score', 'episodes'])
    
    async def test_load_data_with_error(self):
        """Test error handling during data loading"""
        # Set up ContentBasedFilter to raise an exception
        self.mock_cbf.side_effect = ValueError("Test error")
        
        # Call the method to test
        await self.engine.load_data()
        
        # Check that the DataFrame was still loaded
        self.assertIsNotNone(self.engine.media_df)
        
        # Check that content_filter is None due to error
        self.assertIsNone(self.engine.content_filter)
    
    async def test_get_recommendations_by_description(self):
        """Test getting recommendations based on description (matching current simple implementation)."""
        # Mock vector DB search results in the raw format expected
        mock_raw_results = [
            {'id': 101, 'distance': 0.9},
            {'id': 102, 'distance': 0.8},
            {'id': 50, 'distance': 0.75}
        ]
        self.mock_vector_db.find_similar_by_description.return_value = mock_raw_results

        # Call the method under test
        query = "epic battle"
        top_n = 3
        results = await self.engine.get_recommendations_by_description(query, k=top_n)
        
        # Assertions
        # Check vector DB call
        self.mock_vector_db.find_similar_by_description.assert_awaited_once_with(query, top_n)

        # Assert final result matches the raw mock output
        self.assertEqual(len(results), 3) 
        self.assertEqual(results, mock_raw_results) # Check if the entire list matches
        self.assertEqual(results[0]['id'], 101)
        self.assertEqual(results[1]['id'], 102)
        self.assertEqual(results[2]['id'], 50)
        self.assertTrue(all('distance' in item for item in results)) # Ensure 'distance' is present

    def test_get_content_based_recommendations(self):
        """Test getting content-based recommendations"""
        # Set up content_filter mock to return results
        mock_content_filter = MagicMock()
        mock_results = [(2, 0.9), (3, 0.8), (5, 0.7)]
        mock_content_filter.get_recommendations.return_value = mock_results
        self.engine.content_filter = mock_content_filter
        
        # Set up engine.media_data and id_to_title mapping
        self.engine.media_data = [
            (1, 'Anime A', 'Description A'),
            (2, 'Anime B', 'Description B'),
            (3, 'Anime C', 'Description C'),
            (4, 'Anime D', 'Description D'),
            (5, 'Anime E', 'Description E')
        ]
        self.engine.id_to_title = {1: 'Anime A', 2: 'Anime B', 3: 'Anime C', 4: 'Anime D', 5: 'Anime E'}
        
        # Call method to test
        item_id = 1
        results = self.engine.get_content_based_recommendations(item_id, k=3)
        
        # Assertions
        mock_content_filter.get_recommendations.assert_called_once_with(item_id, top_n=3)
        self.assertEqual(len(results), 3)
        # Check format of results (id, title, score)
        self.assertEqual(results[0][0], 2)
        self.assertEqual(results[0][1], 'Anime B')
        self.assertEqual(results[0][2], 0.9)
    
    def test_get_item_details(self):
        """Test getting details for a specific item"""
        # Call the method
        item_id = 1
        result = self.engine.get_item_details(item_id)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['anime_id'], 1)
        self.assertEqual(result['title'], 'Anime A')
        self.assertEqual(result['type'], 'tv')
        self.assertEqual(result['score'], 8.5)
        
        # Test with non-existent ID
        result = self.engine.get_item_details(999)
        self.assertEqual(result, {})
    
    async def test_fallback_content_filter_initialization(self):
        """Test the fallback mechanism for ContentBasedFilter initialization"""
        # First call raises error, second call succeeds
        mock_cbf_first = MagicMock()
        mock_cbf_second = MagicMock()
        self.mock_cbf.side_effect = [ValueError("Int64 dtype error"), mock_cbf_second]
        
        # Mock the print function to check outputs
        with patch('builtins.print') as mock_print:
            await self.engine.load_data()
            
            # Check that it tried with numeric features first, then without
            # First call should have both text and numeric features
            args1, kwargs1 = self.mock_cbf.call_args_list[0]
            self.assertIn('numeric_feature_cols', kwargs1)
            self.assertEqual(kwargs1['numeric_feature_cols'], ['score', 'episodes'])
            
            # Second call should only have text features
            args2, kwargs2 = self.mock_cbf.call_args_list[1]
            self.assertIn('numeric_feature_cols', kwargs2)
            self.assertEqual(kwargs2['numeric_feature_cols'], [])
            
            # Check we got the expected success message
            mock_print.assert_any_call("ContentBasedFilter initialized with text features only.")


if __name__ == '__main__':
    unittest.main()
