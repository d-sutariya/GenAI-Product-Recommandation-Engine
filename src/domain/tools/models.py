from typing import Any, Dict, Union, List
from pydantic import BaseModel

class ToolCallResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Union[str, list, dict]
    raw_response: Any = None