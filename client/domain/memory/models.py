from typing import List, Optional
from pydantic import BaseModel

class MemoryRecord(BaseModel):
    """Domain entity for a memory record."""
    text: str
    type: str = "fact"
    timestamp: Optional[str] = None
    tool_name: Optional[str] = None
    user_query: Optional[str] = None
    tags: List[str] = []
    user_id: Optional[str] = None
    session_id: Optional[str] = None