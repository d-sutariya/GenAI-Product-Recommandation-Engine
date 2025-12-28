from typing import TypedDict, List, Optional, Any
from domain.memory.memory_port import MemoryRecord, MemoryManager
from src.domain.perception.models import PerceptionResult

class AgentState(TypedDict):
    """
    State definition for the RAG agent graph.
    This state is passed between nodes and updated throughout execution.
    """
    # User input
    user_input: str
    original_query: str
    session_id: str
    
    # Cognitive layers
    perception: Optional[PerceptionResult]
    memory_items: List[MemoryRecord]  # Retrieved memories for current query (replaced each iteration)
    decision: str
    tool_result: Optional[ToolCallResult]
    
    # MCP context (passed through state but not modified by LangGraph)
    mcp_session: ClientSession
    mcp_tools: List[Any]
    tool_descriptions: str
    memory: MemoryManager
    
    # Control flow
    step: int
    max_steps: int
    final_answer: Optional[str]
    error: Optional[str]
    should_continue: bool

