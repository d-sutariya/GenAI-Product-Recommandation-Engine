from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class DecisionResult(BaseModel):
    thought: str = Field(
        description="Your internal reasoning process. Explain why you are choosing this tool or providing this answer."
    )
    decision_type: Literal["tool_call", "final_answer"] = Field(
        description="The type of action you are taking. 'tool_call' if you need more information, 'final_answer' if you have the answer."
    )
    tool_name: Optional[str] = Field(
        default=None, 
        description="The exact name of the tool to call. Required if decision_type is 'tool_call'."
    )
    tool_input: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="The arguments for the tool call. Required if decision_type is 'tool_call'."
    )
    final_answer: Optional[str] = Field(
        default=None, 
        description="The natural language response to the user. Required if decision_type is 'final_answer'."
    )
    recommended_product: Optional[str] = Field(
        default=None,
        description="The name of the product being recommended, if any. Used to trigger add-to-cart logic."
    )
