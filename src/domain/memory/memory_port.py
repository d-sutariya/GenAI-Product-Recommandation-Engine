from abc import ABC, abstractmethod
from typing import List, Optional, Any
from src.domain.memory.models import MemoryRecord

class MemoryStore(ABC):
    @abstractmethod
    def add(self, item: MemoryRecord) -> None:
        """Add a single item to memory."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3, session_filter: Optional[str] = None) -> List[MemoryRecord]:
        """Retrieve items from memory."""
        pass
