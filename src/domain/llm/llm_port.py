from abc import ABC, abstractmethod
from pydantic import BaseModel

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates text from a prompt."""
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: BaseModel) -> BaseModel:
        """Generates structured data (like JSON or Pydantic) from a prompt."""
        pass
