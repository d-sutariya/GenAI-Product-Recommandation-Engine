from typing import Optional, List
from pydantic import Field, BaseModel

class PerceptionResult(BaseModel):
    user_input: str = Field(description="The original user input")
    modified_user_input: Optional[str] = Field(default=None, description="A modified version of the user input, if applicable")
    intent: Optional[str] = Field(default=None, description="The inferred intent of the user")
    entities: List[str] = Field(default_factory=list, description="A list of extracted entities or attributes")
    tool_hint: Optional[str] = Field(default=None, description="A hint for which tool to use, if applicable")
