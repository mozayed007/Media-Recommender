import unittest
import asyncio
import os
import sys
import numpy as np
from typing import List, Optional, Union, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vector_database import LlamaIndexVectorDatabase, create_vector_database
from src.abstract_interface_classes import AbstractEmbeddingModel

class MockEmbeddingModel(AbstractEmbeddingModel):
    async def get_text_embedding(self, text: str) -> List[float]:
        return [0.1] * 128  # Return a mock embedding of dimension 128

    async def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 128 for _ in texts]

class TestLlamaIndexVectorDatabase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = os.path.join(os.path.dirname(__file__), "test_llama_index_db")
        cls.embedding_model = MockEmbeddingModel()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db_path):
            import shutil
            shutil.rmtree(cls.test_db_path)

    async def asyncSetUp(self):
        self.db = await create_vector_database(
            self.embedding_model,
            collection_name="test_collection",
            db_path=self.test_db_path
        )

    async def asyncTearDown(self):
        if hasattr(self, 'db'):
            await self.db.clear()
            await self.db.close()

    async def test_create_vector_database(self):
        self.assertIsInstance(self.db, LlamaIndexVectorDatabase)
        self.assertIsNotNone(self.db.index)

    async def test_add_and_retrieve_items(self):
        # Add items
        await self.db.add_items([1, 2, 3], ["Title1", "Title2", "Title3"], ["Desc1", "Desc2", "Desc3"])
        
        # Retrieve by description
        results = await self.db.find_similar_by_description("Test query", k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 3)  # (id, title, distance)
        
        # Retrieve by title
        results = await self.db.find_similar_by_title("Test title", k=2)
        self.assertEqual(len(results), 2)

    async def test_update_and_get_item(self):
        # Add an item
        await self.db.add_items([1], ["Original Title"], ["Original Description"])
        
        # Update the item
        await self.db.update(1, "Updated Title", "Updated Description")
        
        # Get the updated item
        item = await self.db.get(1)
        self.assertIsNotNone(item)
        self.assertEqual(item['media_id'], 1)
        self.assertEqual(item['title'], "Updated Title")
        self.assertEqual(item['description'], "Updated Description")

    async def test_count_and_clear(self):
        # Add items
        await self.db.add_items(
            [1, 2, 3, 4, 5],
            ["The Matrix", "Inception", "Interstellar", "Blade Runner", "Ex Machina"],
            [
                "A computer programmer discovers a dystopian world ruled by machines.",
                "A thief who enters people's dreams to steal secrets is given a chance at redemption.",
                "A team of explorers travel through a wormhole in space to ensure humanity's survival.",
                "A blade runner must pursue and terminate four replicants who have returned to Earth.",
                "A programmer participates in an experiment to evaluate the human qualities of a highly advanced humanoid AI."
            ]
        )
        
        # Check count
        count = await self.db.count()
        self.assertEqual(count, 5)
        
        # Clear the database
        await self.db.clear()
        
        # Check count after clearing
        count = await self.db.count()
        self.assertEqual(count, 0)

def run_tests_individually():
    test_cases = [
        'test_create_vector_database',
        'test_add_and_retrieve_items',
        'test_update_and_get_item',
        'test_count_and_clear'
    ]
    
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestLlamaIndexVectorDatabase)
    
    for test_name in test_names:
        if test_name in test_cases:
            suite = unittest.TestSuite()
            suite.addTest(TestLlamaIndexVectorDatabase(test_name))
            
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            if not result.wasSuccessful():
                print(f"Test '{test_name}' failed. Exiting.")
                sys.exit(1)
            
            print(f"Test '{test_name}' passed.")
            print("---")

if __name__ == '__main__':
    asyncio.run(run_tests_individually())