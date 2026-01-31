import asyncio
import time
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from client.utils.logger import log
from client.domain.shared.state import AgentState
from client.infrastructure.llm.gemini_adapter import GeminiLLMAdapter
from client.infrastructure.llm.huggingface_adapter import HFLLMAdapter
from client.infrastructure.memory.faiss_memory_adapter import FaissMemoryAdapter
from client.infrastructure.tools.mcp_tool_adapter import MCPToolAdapter
from client.application.services.perception import PerceptionService
from client.application.services.reasoning import DecisionService
from client.application.services.agent_orchestrator import AgentWorkflow
from client.application.services.client_history_rag import ClientHistoryRAGService

async def main(user_input: str):
    """
    Main agent function using Enterprise Hexagonal Architecture.
    """
    try:
        log("agent", "Starting Product Recommendation Agent...")
        
        # 1. Initialize Adapters (Infrastructure)
        # LLM
        try:
            llm_adapter = HFLLMAdapter()
        except Exception as e:
            log("error", f"Failed to init LLM: {e}")
            return

        # Memory
        memory_adapter = FaissMemoryAdapter()

        # MCP Server Params
        server_params = StdioServerParameters(
            command="C:\\Users\\DELL\\miniconda3\\Scripts\\uv.EXE",
            args=[
                "C:\\Users\\DELL\\Desktop\\project\\rag_mcp_product_recommandation\\RAG_MCP\\server\\mcp_server.py"
			],
            cwd="."
        )

        # 2. Connect to MCP (Tool Adapter)
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Tool Adapter
                    tool_adapter = MCPToolAdapter(session)
                    
                    # Pre-fetch tools for descriptions
                    tools = await tool_adapter.list_tools()
                    tool_descriptions = tool_adapter.get_tool_descriptions()
                    log("agent", f"Loaded tools: {len(tools)}")

                    # 3. Initialize Domain Services (Business Logic)
                    perception_service = PerceptionService(llm_adapter)
                    decision_service = DecisionService(llm_adapter)

                    # 4. Initialize Application Workflow (Wiring)
                    workflow_app = AgentWorkflow(
                        perception_service=perception_service,
                        decision_service=decision_service,
                        memory_store=memory_adapter,
                        tool_executor=tool_adapter
                    )

                    rag_service = ClientHistoryRAGService(memory_adapter, llm_adapter)
                    user_id = "test_user_01"  # Hardcoded for CLI demo, auth is yet to implemented
                    
                    app = workflow_app.build()

                    # 5. Execute
                    session_id = f"session-{int(time.time())}"
                    initial_state: AgentState = {
                        "user_input": user_input,
                        "original_query": user_input,
                        "session_id": session_id,
                        "user_id": user_id,
                        "perception": None,
                        "memory_items": [],
                        "decision": "",
                        "tool_result": None,
                        "tool_descriptions": tool_descriptions,
                        "step": 0,
                        "max_steps": 5,
                        "final_answer": None,
                        "error": None,
                        "should_continue": True
                    }

                    log("agent", "Starting graph execution...")
                    final_state = await app.ainvoke(initial_state)

                    final_answer = final_state.get("final_answer")
                    if final_answer:
                        log("agent", f"FINAL RESULT: {final_answer}")
                        print("\n" + final_answer.replace("FINAL_ANSWER:", "").strip() + "\n")
                    else:
                        log("agent", "No final answer generated")
                        print("No final answer generated.")
                    
                    if final_answer:
                        rag_service.add_interaction(user_id, user_input, final_answer.replace("FINAL_ANSWER:", "").strip(), session_id)

        except Exception as e:
            log("error", f"Fatal error: {str(e)}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        log("error", f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = input("What Product do you want to find Today? → ")
    asyncio.run(main(query))


