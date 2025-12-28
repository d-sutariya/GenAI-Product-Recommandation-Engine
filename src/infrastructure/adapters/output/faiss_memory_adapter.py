import faiss
import numpy as np
import os
from typing import List, Optional
from dotenv import load_dotenv
from src.application.ports.output.memory_port import MemoryStore, MemoryRecord
from src.infrastructure.adapters.output.gemini_adapter import GeminiLLMAdapter
from google import genai

load_dotenv()

class FaissMemoryAdapter(MemoryStore):
    def __init__(self, embedding_model: str = "text-embedding-004"):
        # We need an embedder. Ideally, Embedder should be its own Port.
        # For simplicity in this refactor, we reuse the Gemini Client structure or separate it.
        # Let's use the GeminiAdapter logic or a lightweight version of it for embeddings.
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.output_dim = 768 # Default for gemini 004 text-embedding
        self.embedding_model = embedding_model
        
        self.index = None
        self.data: List[MemoryRecord] = []
        self.embeddings: List[np.ndarray] = []

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            response = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                content=text
            )
            # Assuming response structure based on original code
            return np.array(response.embeddings[0].values, dtype=np.float32)
        except Exception as e:
            print(f"Failed to get embeddings: {e}")
            raise

    def add(self, item: MemoryRecord) -> None:
        emb = self._get_embedding(item.text)
        self.embeddings.append(emb)
        self.data.append(item)

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(emb))
        self.index.add(np.stack([emb]))

    def retrieve(self, query: str, top_k: int = 3, session_filter: Optional[str] = None) -> List[MemoryRecord]:
        if self.index is None or len(self.data) == 0:
            return []

        query_vec = self._get_embedding(query).reshape(1, -1)
        D, I = self.index.search(query_vec, top_k * 2) # Overfetch

        results = []
        for idx in I[0]:
            if idx >= len(self.data):
                continue
            item = self.data[idx]

            if session_filter and item.session_id != session_filter:
                continue

            results.append(item)
            if len(results) >= top_k:
                break
        
        return results
