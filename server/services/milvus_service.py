"""
Milvus Lite database service for vector operations
"""

from pymilvus import MilvusClient
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

from config.settings import settings
from utils.logger import logger


class MilvusService:
    """Service for managing Milvus Lite vector database operations"""
    
    def __init__(self):
        """Initialize Milvus Lite connection"""
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dimension = settings.MILVUS_DIMENSION
        self.db_file = settings.ROOT_DIR / "milvus_lite" / "products.db"
        self.db_file.parent.mkdir(exist_ok=True)
        self.client: MilvusClient = None
        self.connect()
        
    def connect(self) -> None:
        """Establish connection to Milvus Lite"""
        try:
            self.client = MilvusClient(str(self.db_file))
            logger.success(f"Connected to Milvus Lite at {self.db_file}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus Lite: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Milvus Lite"""
        try:
            if self.client:
                self.client.close()
            logger.info("Disconnected from Milvus Lite")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus Lite: {e}")
    
    def create_collection(self) -> None:
        """Create product collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.list_collections()
            
            if self.collection_name in collections:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            
            logger.success(f"Created collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def create_index(self) -> None:
        """Create index on embedding field for efficient search"""
        raise NotImplementedError("Index creation is not yet implemented for Milvus Lite")
    
    def insert_data(
        self,
        ids: List[int],
        product_ids: List[int],
        product_contents: List[str],
        metadatas: List[str],
        embeddings: List[np.ndarray]
    ) -> None:
        """
        Insert product data into Milvus Lite collection
        
        Args:
            ids: Unique IDs for each record
            product_ids: Product IDs
            product_contents: Product content strings
            metadatas: Product metadata as JSON strings
            embeddings: Product embeddings
        """
        try:
            # Convert embeddings to list format
            embedding_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
            
            # Prepare data for insertion
            data = [
                {
                    "id": ids[i],
                    "vector": embedding_list[i],
                    "product_id": product_ids[i],
                    "product_content": product_contents[i],
                    "metadata": metadatas[i]
                }
                for i in range(len(ids))
            ]
            
            # Insert data
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            logger.success(f"Inserted {result.get('insert_count', len(ids))} records into collection")
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar products
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results with product information
        """
        try:
            query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["product_id", "product_content", "metadata"]
            )
            
            # Format results
            formatted_results = []
            if results and len(results) > 0:
                for result in results[0]:
                    formatted_results.append({
                        "id": result.get("entity", {}).get("product_id"),
                        "product_content": result.get("entity", {}).get("product_content"),
                        "metadata": result.get("entity", {}).get("metadata"),
                        "distance": result.get("distance", 0)
                    })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def count_entities(self) -> int:
        """Get number of entities in collection"""
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return stats.get('row_count', 0)
        except Exception as e:
            logger.error(f"Failed to count entities: {e}")
            return 0
    
    def drop_collection(self) -> None:
        """Drop the collection (use with caution)"""
        try:
            self.client.drop_collection(collection_name=self.collection_name)
            logger.warn(f"Dropped collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")


# Global milvus service instance
milvus_service = MilvusService()
