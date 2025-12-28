from abc import ABC, abstractmethod
from typing import List, Optional, Any
from pydantic import BaseModel

class MemoryRecord(BaseModel):
    """Domain entity for a memory record."""
    text: str
    type: str = "fact"
    timestamp: Optional[str] = None
    tool_name: Optional[str] = None
    user_query: Optional[str] = None
    tags: List[str] = []
    session_id: Optional[str] = None

class MemoryStore(ABC):
    @abstractmethod
    def add(self, item: MemoryRecord) -> None:
        """Add a single item to memory."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3, session_filter: Optional[str] = None) -> List[MemoryRecord]:
        """Retrieve items from memory."""
        pass
