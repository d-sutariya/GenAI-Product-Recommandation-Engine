from typing import Any, Dict, List
import ast
from mcp import ClientSession

from src.application.ports.output.tool_port import ToolExecutor, ToolCallResult
from src.utils.logger import log

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

    def _parse_function_call(self, response: str) -> tuple[str, Dict[str, Any]]:
        """Helpers method to parse the string response"""
        try:
            if not response.startswith("FUNCTION_CALL:"):
                # Fallback or error
                raise ValueError("Not a valid FUNCTION_CALL")

            _, function_info = response.split(":", 1)
            parts = [p.strip() for p in function_info.split("|")]
            if len(parts) == 1:
                func_name = parts[0]
                param_parts = []
            else:
                func_name, param_parts = parts[0], parts[1:]

            result = {}
            for part in param_parts:
                if "=" not in part:
                     # Simple error handling for malformed params
                    continue
                key, value = part.split("=", 1)
                try:
                    parsed_value = ast.literal_eval(value)
                except Exception:
                    parsed_value = value.strip()

                keys = key.split(".")
                current = result
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = parsed_value

            return func_name, result
        except Exception as e:
            log("parser", f"Failed to parse: {e}")
            raise

    async def execute(self, tool_name: str, arguments: Dict[str, Any] = None) -> ToolCallResult:
        # In the original code, the controller parsed the string.
        # In Hexagonal, the Adapter should handle the Low-Level mechanics.
        # BUT, the `execute` signature here expects tool_name and args.
        # This means the Application Layer (Graph) should probably do the parsing OR we ask the Port to parse.
        # Let's assume the Domain/App layer passes specific args, OR the raw string if we want to Encapsulate Parsing here.
        
        # Correction: The original code had `execute_tool(session, tools, response_string)`.
        # To strictly follow the Port definition `execute(tool_name, args)`, parsing should happen BEFORE calling this, 
        # OR we overload this method.
        # However, for this specific "LLM Output -> Tool Call" pattern, let's keep it simple: 
        # The Application Layer will use a helper (or this adapter) to parse, then call execute.
        
        # Wait, I'll add a specific method for handling the raw string execution to makes things easier for the graph
        pass 

    async def execute_raw(self, response_string: str) -> ToolCallResult:
        """Executes a tool based on the raw LLM output string."""
        tool_name, args = self._parse_function_call(response_string)
        log("tool", f"⚙️ Calling '{tool_name}' with: {args}")
        
        result = await self.session.call_tool(tool_name, arguments=args)
        
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
            arguments=args,
            result=out,
            raw_response=result
        )

    # Implementing the interface method
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        # Direct execution if we already parsed it
        return await self.execute_raw(f"FUNCTION_CALL: {tool_name}|args={arguments}") # Hacky re-serialization? 
        # Better: just implement direct call:
        log("tool", f"⚙️ Calling '{tool_name}' with: {arguments}")
        result = await self.session.call_tool(tool_name, arguments=arguments)
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
