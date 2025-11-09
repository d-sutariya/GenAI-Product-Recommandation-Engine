"""
LangGraph nodes for the RAG agent.
Each node represents a cognitive layer or action in the agent flow.
"""
from state import AgentState
from perception import extract_perception
from memory import MemoryRecord
from decision import generate_plan
from action import execute_tool
from log_utils import log
import json


def perception_node(state: AgentState) -> AgentState:
    """
    Perception node - extracts structured information from user input.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with perception result
    """
    log("perception", "Starting perception extraction...")
    
    user_input = state["user_input"]
    perception = extract_perception(user_input)
    
    state["perception"] = perception
    perception_result_str = json.dumps(perception.model_dump(), indent=4)
    log("perception", f"Perception result: {perception_result_str}")
    
    return state


def memory_node(state: AgentState) -> AgentState:
    """
    Memory node - retrieves relevant past interactions.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with retrieved memories
    """
    log("memory", "Retrieving relevant memories...")
    
    memory = state["memory"]
    session_id = state["session_id"]
    query = state["user_input"]
    
    retrieved = memory.retrieve(
        query=query,
        top_k=3,
        session_filter=session_id
    )
    
    # Return only new memories (LangGraph will append to existing list via operator.add)
    # We return a list, and LangGraph will add it to the existing memory_items
    state["memory_items"] = retrieved
    log("memory", f"Retrieved {len(retrieved)} relevant memories")
    
    return state


def decision_node(state: AgentState) -> AgentState:
    """
    Decision node - generates plan (tool call or final answer) using LLM.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with decision/plan
    """
    log("decision", "Generating decision/plan...")
    
    perception = state["perception"]
    memory_items = state["memory_items"]
    tool_descriptions = state["tool_descriptions"]
    
    plan = generate_plan(
        perception=perception,
        memory_items=memory_items,
        tool_descriptions=tool_descriptions
    )
    
    state["decision"] = plan
    
    # If it's a final answer, set it in state for routing
    if plan.startswith("FINAL_ANSWER:"):
        state["final_answer"] = plan
        state["should_continue"] = False
    
    log("decision", f"Plan generated: {plan}")
    
    return state


async def mcp_tool_execution_node(state: AgentState) -> AgentState:
    """
    MCP Tool Execution node - executes tools via MCP protocol.
    This is a custom node that wraps the existing action.py logic.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with tool execution result
    """
    log("tool", "Executing tool via MCP...")
    
    decision = state["decision"]
    session = state["mcp_session"]
    tools = state["mcp_tools"]
    
    # Check if it's a final answer (shouldn't reach here, but safety check)
    if decision.startswith("FINAL_ANSWER:"):
        state["final_answer"] = decision
        state["should_continue"] = False
        return state
    
    # Execute tool via MCP using existing action.py logic
    try:
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        tool_result = await execute_tool(
            session=session,
            tools=tools,
            response=decision
        )
        
        state["tool_result"] = tool_result
        state["error"] = None
        log("tool", f"{tool_result.tool_name} returned: {tool_result.result}")
        
    except Exception as e:
        state["error"] = str(e)
        state["tool_result"] = None
        log("error", f"Tool execution failed: {e}")
    
    return state


def memory_update_node(state: AgentState) -> AgentState:
    """
    Memory Update node - stores tool execution results in memory.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with memory updated and user_input for next iteration
    """
    log("memory", "Updating memory with tool result...")
    
    memory = state["memory"]
    tool_result = state.get("tool_result")
    session_id = state["session_id"]
    user_input = state["user_input"]
    original_query = state["original_query"]
    error = state.get("error")
    
    if tool_result and not error:
        # Add tool result to memory using MemoryManager
        memory.add(MemoryRecord(
            text=f"Tool call: {tool_result.tool_name} with {tool_result.arguments}, got: {tool_result.result}",
            type="tool_output",
            session_id=session_id,
            user_query=user_input,
            tags=[tool_result.tool_name],
            tool_name=tool_result.tool_name
        ))
        
        # Update user input for next iteration
        state["user_input"] = (
            f"Original task: {original_query}\n"
            f"Previous output: {tool_result.result}\n"
            f"What should I do next?"
        )
        
        # Increment step
        state["step"] = state["step"] + 1
        log("memory", "Memory updated successfully")
    else:
        log("memory", "No tool result to store (error or final answer)")
    
    return state


def route_decision(state: AgentState) -> str:
    """
    Route function for conditional edge from decision node.
    Determines whether to execute a tool or end with final answer.
    This is a pure routing function - it only reads state and returns route.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node to execute: "tool" or "end"
    """
    decision = state["decision"]
    final_answer = state.get("final_answer")
    
    # If final answer is already set in decision_node, route to end
    if final_answer:
        log("router", "Routing to END (final answer)")
        return "end"
    elif decision.startswith("FINAL_ANSWER:"):
        log("router", "Routing to END (final answer in decision)")
        return "end"
    elif decision.startswith("FUNCTION_CALL:"):
        log("router", "Routing to tool execution")
        return "tool"
    else:
        # Default to end if unclear
        log("router", "Routing to END (default/unclear decision)")
        return "end"


def check_continue(state: AgentState) -> str:
    """
    Check whether to continue the loop or end.
    
    Args:
        state: Current agent state
        
    Returns:
        Next action: "continue", "end", or "error"
    """
    # Check for final answer
    if state.get("final_answer"):
        log("router", "Ending: final answer reached")
        return "end"
    
    # Check for error
    if state.get("error"):
        log("router", "Ending: error occurred")
        return "error"
    
    # Check max steps
    if state["step"] >= state["max_steps"]:
        log("router", "Ending: max steps reached")
        state["final_answer"] = "FINAL_ANSWER: [Max steps reached. Please refine your query.]"
        return "end"
    
    # Continue loop
    log("router", "Continuing to next iteration")
    state["should_continue"] = True
    return "continue"


def error_handler_node(state: AgentState) -> AgentState:
    """
    Error handler node - handles errors gracefully.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with error information
    """
    error = state.get("error", "Unknown error")
    log("error", f"Error handler: {error}")
    
    state["final_answer"] = f"FINAL_ANSWER: [Error occurred: {error}]"
    state["should_continue"] = False
    
    return state

