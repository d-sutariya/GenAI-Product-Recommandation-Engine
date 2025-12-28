from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates text from a prompt."""
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Any) -> Any:
        """Generates structured data (like JSON or Pydantic) from a prompt."""
        pass
