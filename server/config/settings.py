"""
Configuration settings for MCP Server
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Paths
    ROOT_DIR: Path = Path(__file__).parent.parent.parent.resolve()
    DOCUMENTS_DIR: Path = ROOT_DIR / "documents"
    
    # Milvus Lite Configuration
    MILVUS_COLLECTION_NAME: str = "product_collection"
    MILVUS_DIMENSION: int = 768  # Dimension for text-embedding-004
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-004"
    
    # MCP Server Configuration
    SERVER_NAME: str = "Product-Recommendation-Agent"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
