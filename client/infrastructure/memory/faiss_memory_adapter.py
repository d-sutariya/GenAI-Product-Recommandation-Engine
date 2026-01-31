import faiss
import numpy as np
import os
from typing import List, Optional
from dotenv import load_dotenv
from client.domain.memory.models import MemoryRecord
from client.domain.memory.memory_port import MemoryStore
from google import genai

load_dotenv()

class FaissMemoryAdapter(MemoryStore):
    def __init__(self, embedding_model: str = "text-embedding-004"):
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.output_dim = 768 # Default for gemini 004 text-embedding
        self.embedding_model = embedding_model
        
        self.index = None
        self.data: List[MemoryRecord] = []
        self.embeddings: List[np.ndarray] = []
        self.index_file = "faiss_index.bin"
        self.data_file = "memory_data.pkl"
        self.load()

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            response = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                contents=[text]
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
        self.save()

    def save(self):
        import pickle
        if self.index:
            faiss.write_index(self.index, self.index_file)
        with open(self.data_file, "wb") as f:
            pickle.dump(self.data, f)
            
    def load(self):
        import pickle
        if os.path.exists(self.index_file) and os.path.exists(self.data_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.data_file, "rb") as f:
                    self.data = pickle.load(f)
                print(f"Loaded {len(self.data)} memory records.")
            except Exception as e:
                print(f"Failed to load memory: {e}")

    def retrieve(self, query: str, top_k: int = 3, session_filter: Optional[str] = None, user_id: Optional[str] = None) -> List[MemoryRecord]:
        if self.index is None or len(self.data) == 0:
            return []

        query_vec = self._get_embedding(query).reshape(1, -1)
        D, I = self.index.search(query_vec, top_k * 5) # Increased overfetch to account for filtering

        results = []
        for idx in I[0]:
            if idx >= len(self.data):
                continue
            item = self.data[idx]

            if session_filter and item.session_id != session_filter:
                continue

            if user_id and item.user_id != user_id:
                continue

            results.append(item)
            if len(results) >= top_k:
                break
        
        return results
