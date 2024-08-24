import unittest
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vector_database import OptimizedMilvusLiteVectorDatabase, create_vector_database
from src.abstract_interface_classes import AbstractEmbeddingModel

class MockEmbeddingModel(AbstractEmbeddingModel):
    async def embed(self, text: str):
        return [0.1] * 128  # Return a mock embedding of dimension 128

    async def embed_batch(self, texts: list):
        return [[0.1] * 128 for _ in texts]

class TestOptimizedMilvusLiteVectorDatabase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_milvus_lite_data.db")
        cls.embedding_model = MockEmbeddingModel()

    @classmethod
    async def tearDown(self):
        connections.disconnect("default")
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    async def test_create_vector_database(self):
        db = await create_vector_database(
            self.embedding_model,
            collection_name="test_collection",
            db_path=self.test_db_path
        )
        self.assertIsInstance(db, OptimizedMilvusLiteVectorDatabase)
        self.assertIsNotNone(db.collection)
        await db.close()


    async def test_add_and_retrieve_items(self):
        db = await create_vector_database(
            self.embedding_model,
            collection_name="test_collection",
            db_path=self.test_db_path
        )
        
        # Add items
        await db.add_items([1, 2, 3], ["Title1", "Title2", "Title3"], ["Desc1", "Desc2", "Desc3"])
        
        # Retrieve by description
        results = await db.find_similar_by_description("Test query", k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 3)  # (id, title, distance)
        
        # Retrieve by title
        results = await db.find_similar_by_title("Test title", k=2)
        self.assertEqual(len(results), 2)
        
        await db.close()

    async def test_update_and_get_item(self):
        db = await create_vector_database(
            self.embedding_model,
            collection_name="test_collection",
            db_path=self.test_db_path
        )
        
        # Add an item
        await db.add_items([1], ["Original Title"], ["Original Description"])
        
        # Update the item
        await db.update(1, "Updated Title", "Updated Description")
        
        # Get the updated item
        item = await db.get(1)
        self.assertIsNotNone(item)
        self.assertEqual(item['media_id'], 1)
        self.assertEqual(item['title'], "Updated Title")
        
        await db.close()

    async def test_count_and_clear(self):
        db = await create_vector_database(
            self.embedding_model,
            collection_name="test_collection",
            db_path=self.test_db_path
        )
        
        # Add items
        await db.add_items([1, 2, 3], ["Title1", "Title2", "Title3"], ["Desc1", "Desc2", "Desc3"])
        
        # Check count
        count = await db.count()
        self.assertEqual(count, 3)
        
        # Clear the database
        await db.clear()
        
        # Check count after clearing
        count = await db.count()
        self.assertEqual(count, 0)
        
        await db.close()

if __name__ == '__main__':
    asyncio.run(unittest.main())