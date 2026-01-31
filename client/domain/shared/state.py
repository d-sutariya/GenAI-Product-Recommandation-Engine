from typing import TypedDict, List, Optional, Any
from client.domain.memory.memory_port import MemoryRecord
from client.domain.perception.models import PerceptionResult
from client.domain.tools.models import ToolCallResult
from client.domain.decision.models import DecisionResult
from mcp import ClientSession
class AgentState(TypedDict):
    """
    State definition for the RAG agent graph.
    This state is passed between nodes and updated throughout execution.
    """
    # User input
    user_input: str
    original_query: str
    user_id: Optional[str]
    session_id: str
    
    # Cognitive layers
    perception: Optional[PerceptionResult]
    memory_items: List[MemoryRecord]  # Retrieved memories 
    decision: Optional[DecisionResult]
    tool_result: Optional[ToolCallResult]
    
    # MCP context
    mcp_session: ClientSession
    mcp_tools: List[Any]
    tool_descriptions: str
    # memory: MemoryManager
    
    # Control flow
    step: int
    max_steps: int
    final_answer: Optional[str]
    error: Optional[str]
    should_continue: bool

