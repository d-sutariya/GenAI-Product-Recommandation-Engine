"""
Milvus database service for vector operations
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import List, Dict, Any, Optional
import numpy as np

from config.settings import settings
from utils.logger import logger


class MilvusService:
    """Service for managing Milvus vector database operations"""
    
    def __init__(self):
        """Initialize Milvus connection"""
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dimension = settings.MILVUS_DIMENSION
        self.collection: Optional[Collection] = None
        
    def connect(self) -> None:
        """Establish connection to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            logger.success(f"Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Milvus server"""
        try:
            connections.disconnect(alias="default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {e}")
    
    def create_collection(self) -> None:
        """Create product collection if it doesn't exist"""
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                self.collection = Collection(self.collection_name)
                return
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="product_id", dtype=DataType.INT64),
                FieldSchema(name="product_content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Product collection with embeddings"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            logger.success(f"Created collection '{self.collection_name}'")
            
            # Create index for vector field
            self.create_index()
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def create_index(self) -> None:
        """Create index on embedding field for efficient search"""
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.success("Created index on embedding field")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def load_collection(self) -> None:
        """Load collection into memory for search"""
        try:
            if self.collection is None:
                self.collection = Collection(self.collection_name)
            
            self.collection.load()
            logger.info(f"Loaded collection '{self.collection_name}' into memory")
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            raise
    
    def insert_data(
        self,
        ids: List[int],
        product_ids: List[int],
        product_contents: List[str],
        metadatas: List[str],
        embeddings: List[np.ndarray]
    ) -> None:
        """
        Insert product data into Milvus collection
        
        Args:
            ids: Unique IDs for each record
            product_ids: Product IDs
            product_contents: Product content strings
            metadatas: Product metadata as JSON strings
            embeddings: Product embeddings
        """
        try:
            if self.collection is None:
                self.collection = Collection(self.collection_name)
            
            # Convert embeddings to list format
            embedding_list = [emb.tolist() for emb in embeddings]
            
            data = [
                ids,
                product_ids,
                product_contents,
                metadatas,
                embedding_list
            ]
            
            self.collection.insert(data)
            self.collection.flush()
            
            logger.success(f"Inserted {len(ids)} records into collection")
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
            if self.collection is None:
                self.collection = Collection(self.collection_name)
            
            # Ensure collection is loaded
            self.load_collection()
            
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["product_id", "product_content", "metadata"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.entity.get("product_id"),
                        "product_content": hit.entity.get("product_content"),
                        "metadata": hit.entity.get("metadata"),
                        "distance": hit.distance
                    })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def count_entities(self) -> int:
        """Get number of entities in collection"""
        try:
            if self.collection is None:
                self.collection = Collection(self.collection_name)
            
            return self.collection.num_entities
        except Exception as e:
            logger.error(f"Failed to count entities: {e}")
            return 0
    
    def drop_collection(self) -> None:
        """Drop the collection (use with caution)"""
        try:
            utility.drop_collection(self.collection_name)
            logger.warn(f"Dropped collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")


# Global milvus service instance
milvus_service = MilvusService()
