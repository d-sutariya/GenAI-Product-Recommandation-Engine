"""
Product ingestion service
Handles loading products from JSON files and ingesting into Milvus
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from config.settings import settings
from utils.logger import logger
from models.products import ProductChunkTyped
from services.embedding_service import embedding_service
from services.milvus_service import milvus_service


class IngestionService:
    """Service for ingesting products into Milvus"""
    
    def __init__(self):
        """Initialize ingestion service"""
        self.documents_dir = settings.DOCUMENTS_DIR
        self.cache_file = settings.ROOT_DIR / "milvus_cache" / "ingestion_cache.json"
        self.cache_file.parent.mkdir(exist_ok=True)
        
    def _load_cache(self) -> Dict[str, str]:
        """Load ingestion cache"""
        if self.cache_file.exists():
            return json.loads(self.cache_file.read_text())
        return {}
    
    def _save_cache(self, cache: Dict[str, str]) -> None:
        """Save ingestion cache"""
        self.cache_file.write_text(json.dumps(cache, indent=2))
    
    def _file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()
    
    def _process_product_file(self, file_path: Path) -> ProductChunkTyped:
        """
        Process a single product JSON file
        
        Args:
            file_path: Path to product JSON file
            
        Returns:
            ProductChunkTyped object
        """
        try:
            product_data = json.loads(file_path.read_text())
            product = ProductChunkTyped.from_json(product_data)
            logger.debug(f"Processed product: {product.id}")
            return product
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            raise
    
    def ingest_products(self, force_reingest: bool = False) -> int:
        """
        Ingest products from JSON files into Milvus
        
        Args:
            force_reingest: Force re-ingestion of all products
            
        Returns:
            Number of products ingested
        """
        logger.info("Starting product ingestion...")
        
        # Load cache
        cache = self._load_cache()
        
        # Find all product JSON files
        product_files = list(self.documents_dir.glob("*.json"))
        
        if not product_files:
            logger.warn(f"No product files found in {self.documents_dir}")
            return 0
        
        logger.info(f"Found {len(product_files)} product files")
        
        # Process files
        products_to_ingest = []
        files_processed = []
        
        for file_path in product_files:
            file_hash = self._file_hash(file_path)
            
            # Skip if already processed and not forcing reingest
            if not force_reingest and file_path.name in cache and cache[file_path.name] == file_hash:
                logger.debug(f"Skipping {file_path.name} - already ingested")
                continue
            
            try:
                product = self._process_product_file(file_path)
                products_to_ingest.append(product)
                files_processed.append((file_path.name, file_hash))
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue
        
        if not products_to_ingest:
            logger.info("No new products to ingest")
            return 0
        
        logger.info(f"Ingesting {len(products_to_ingest)} products...")
        
        # Generate embeddings
        embeddings = []
        for product in products_to_ingest:
            embedding = embedding_service.get_embedding(product.product_content)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.error(f"Failed to generate embedding for product {product.id}")
                return 0
        
        # Prepare data for Milvus
        ids = [i for i in range(milvus_service.count_entities(), 
                                 milvus_service.count_entities() + len(products_to_ingest))]
        product_ids = [p.id for p in products_to_ingest]
        product_contents = [p.product_content for p in products_to_ingest]
        metadatas = [p.metadata.model_dump_json() for p in products_to_ingest]
        
        # Insert into Milvus
        milvus_service.insert_data(
            ids=ids,
            product_ids=product_ids,
            product_contents=product_contents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        # Update cache
        for file_name, file_hash in files_processed:
            cache[file_name] = file_hash
        self._save_cache(cache)
        
        logger.success(f"Successfully ingested {len(products_to_ingest)} products")
        return len(products_to_ingest)
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """
        Get current ingestion status
        
        Returns:
            Dictionary with ingestion statistics
        """
        cache = self._load_cache()
        total_files = len(list(self.documents_dir.glob("*.json")))
        ingested_files = len(cache)
        entities_count = milvus_service.count_entities()
        
        return {
            "total_files": total_files,
            "ingested_files": ingested_files,
            "entities_in_milvus": entities_count,
            "pending_files": total_files - ingested_files
        }


# Global ingestion service instance
ingestion_service = IngestionService()
