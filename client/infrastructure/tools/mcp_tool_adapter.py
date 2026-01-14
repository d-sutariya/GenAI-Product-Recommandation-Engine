from typing import Any, Dict, List
import ast
from mcp import ClientSession

from client.domain.tools.tool_port import ToolExecutor, ToolCallResult
from client.utils.logger import log

class MCPToolAdapter(ToolExecutor):
    def __init__(self, session: ClientSession):
        self.session = session
        self.cached_tools = None

    async def list_tools(self) -> List[Any]:
        if not self.cached_tools:
            result = await self.session.list_tools()
            self.cached_tools = result.tools
        return self.cached_tools

    def get_tool_descriptions(self) -> str:
        if not self.cached_tools:
            return ""
        return "\n".join(
            f"- {tool.name}: {getattr(tool, 'description', 'No description')}"
            for tool in self.cached_tools
        )

    async def execute(self, tool_name: str, arguments: Dict[str, Any] = None) -> ToolCallResult:
        log("tool", f"Calling '{tool_name}' with: {arguments}")
        
        result = await self.session.call_tool(tool_name, arguments=arguments)
        
        # Formatting result
        if hasattr(result, 'content'):
            if isinstance(result.content, list):
                out = [getattr(item, 'text', str(item)) for item in result.content]
            else:
                out = getattr(result.content, 'text', str(result.content))
        else:
            out = str(result)

        return ToolCallResult(
            tool_name=tool_name,
            arguments=arguments,
            result=out,
            raw_response=result
        )
