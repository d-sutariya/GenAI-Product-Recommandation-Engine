"""
Embedding service using Google Gemini
"""

import numpy as np
from google import genai
from typing import List, Optional

from config.settings import settings
from utils.logger import logger


class EmbeddingService:
    """Service for generating embeddings using Google Gemini"""
    
    def __init__(self):
        """Initialize Gemini client"""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")
            
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        logger.info(f"Initialized EmbeddingService with model: {self.model}")
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=[text]
            )
            embedding = np.array(response.embeddings[0].values, dtype=np.float32)
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of numpy arrays of embeddings
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings


# Global embedding service instance
embedding_service = EmbeddingService()
