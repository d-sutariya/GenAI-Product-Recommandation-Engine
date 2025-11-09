"""
Main agent entry point using LangGraph for orchestration.
Replaces the manual loop in agent.py with LangGraph state graph.
"""
import asyncio
import time
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from log_utils import log
from state import AgentState
from graph import create_agent_graph
from memory import MemoryManager


async def main(user_input: str):
    """
    Main agent function using LangGraph orchestration.
    
    Args:
        user_input: User's product search query
    """
    try:
        log("agent", "Starting LangGraph-based agent...")
        log("agent", f"Current working directory: {os.getcwd()}")
        
        # Initialize MCP server connection
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            cwd="."
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                log("agent", "Connection established, creating session...")
                
                try:
                    async with ClientSession(read, write) as session:
                        log("agent", "Session created, initializing...")
                        
                        try:
                            # Initialize MCP session
                            await session.initialize()
                            log("agent", "MCP session initialized")
                            
                            # Get available tools
                            tools_result = await session.list_tools()
                            tools = tools_result.tools
                            tool_descriptions = "\n".join(
                                f"- {tool.name}: {getattr(tool, 'description', 'No description')}"
                                for tool in tools
                            )
                            log("agent", f"{len(tools)} tools loaded")
                            
                            # Initialize memory
                            memory = MemoryManager()
                            session_id = f"session-{int(time.time())}"
                            
                            # Create LangGraph
                            app = create_agent_graph()
                            
                            # Prepare initial state
                            initial_state: AgentState = {
                                "user_input": user_input,
                                "original_query": user_input,
                                "session_id": session_id,
                                "perception": None,
                                "memory_items": [],
                                "decision": "",
                                "tool_result": None,
                                "mcp_session": session,
                                "mcp_tools": tools,
                                "tool_descriptions": tool_descriptions,
                                "memory": memory,
                                "step": 0,
                                "max_steps": 5,
                                "final_answer": None,
                                "error": None,
                                "should_continue": True
                            }
                            
                            # Run the graph
                            # Use checkpointing config if available, otherwise use empty config
                            try:
                                config = {"configurable": {"thread_id": session_id}}
                            except Exception:
                                config = {}
                            
                            log("agent", "Starting graph execution...")
                            
                            final_state = await app.ainvoke(initial_state, config)
                            
                            # Extract final answer
                            final_answer = final_state.get("final_answer")
                            if final_answer:
                                log("agent", f"✅ FINAL RESULT: {final_answer}")
                                # Print final answer (remove FINAL_ANSWER: prefix if present)
                                if final_answer.startswith("FINAL_ANSWER:"):
                                    print(final_answer.replace("FINAL_ANSWER:", "").strip())
                                else:
                                    print(final_answer)
                            else:
                                log("agent", "⚠️ No final answer generated")
                                print("No final answer generated. Please check the logs.")
                            
                        except Exception as e:
                            log("error", f"Session initialization error: {str(e)}")
                            print(f"Error: {str(e)}")
                            
                except Exception as e:
                    log("error", f"Session creation error: {str(e)}")
                    print(f"Error: {str(e)}")
                    
        except Exception as e:
            log("error", f"MCP connection error: {e}")
            print(f"Error: {e}")
            sys.exit(1)
            
    except Exception as e:
        log("error", f"Overall error: {str(e)}")
        print(f"Error: {str(e)}")
    
    log("agent", "Agent session complete.")


if __name__ == "__main__":
    query = input("🧑 What Product do you want to find Today? → ")
    asyncio.run(main(query))

