"""
LangGraph graph construction and compilation.
Creates the state graph that orchestrates the RAG agent flow.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from nodes import (
    perception_node,
    memory_node,
    decision_node,
    mcp_tool_execution_node,
    memory_update_node,
    route_decision,
    check_continue,
    error_handler_node
)
from log_utils import log


def create_agent_graph():
    """
    Create and compile the LangGraph state graph for the RAG agent.
    
    Returns:
        Compiled LangGraph application
    """
    log("graph", "Building LangGraph state graph...")
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("perception", perception_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("mcp_tool_execution", mcp_tool_execution_node)
    workflow.add_node("memory_update", memory_update_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Set entry point
    workflow.set_entry_point("perception")
    
    # Add edges
    workflow.add_edge("perception", "memory")
    workflow.add_edge("memory", "decision")
    
    # Conditional edge from decision - route based on decision type
    workflow.add_conditional_edges(
        "decision",
        route_decision,
        {
            "tool": "mcp_tool_execution",
            "end": END
        }
    )
    
    # Edge from tool execution to memory update
    workflow.add_edge("mcp_tool_execution", "memory_update")
    
    # Conditional edge from memory update - check if should continue
    workflow.add_conditional_edges(
        "memory_update",
        check_continue,
        {
            "continue": "perception",  # Loop back to perception
            "end": END,
            "error": "error_handler"
        }
    )
    
    # Edge from error handler to end
    workflow.add_edge("error_handler", END)
    
    # Compile the graph
    # Note: Checkpointing can be added later if needed for state persistence
    # For now, we compile without checkpointing for simplicity
    try:
        memory_saver = MemorySaver()
        app = workflow.compile(checkpointer=memory_saver)
    except Exception:
        # Fallback: compile without checkpointing if there are issues
        log("graph", "Warning: Compiling without checkpointing")
        app = workflow.compile()
    
    log("graph", "LangGraph compiled successfully")
    
    return app

