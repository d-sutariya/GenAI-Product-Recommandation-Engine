from abc import ABC, abstractmethod
from typing import Any, Dict, List
from src.domain.tools.models import ToolCallResult
class ToolExecutor(ABC):
    @abstractmethod
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Executes a tool call."""
        pass

    @abstractmethod
    async def list_tools(self) -> List[Any]:
        """Lists available tools."""
        pass
        
    @abstractmethod
    def get_tool_descriptions(self) -> str:
        """Returns string description of tools."""
        pass
