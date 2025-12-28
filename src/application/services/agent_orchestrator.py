from langgraph.graph import StateGraph, END
from src.domain.models.state import AgentState
from src.application.services.perception_logic import PerceptionService
from src.application.services.reasoning_logic import DecisionService
from src.domain.ports.output.memory_port import MemoryStore, MemoryRecord
from src.domain.ports.output.tool_port import ToolExecutor
from google import genai
from src.utils.logger import log
import json

class AgentWorkflow:
    def __init__(
        self,
        perception_service: PerceptionService,
        decision_service: DecisionService,
        memory_store: MemoryStore,
        tool_executor: ToolExecutor
    ):
        self.perception_service = perception_service
        self.decision_service = decision_service
        self.memory_store = memory_store
        self.tool_executor = tool_executor

    def _perception_node(self, state: AgentState) -> AgentState:
        log("perception", "Starting perception extraction...")
        user_input = state["user_input"]
        perception = self.perception_service.analyze_input(user_input)
        state["perception"] = perception
        # log("perception", f"Perception: {perception}")
        return state

    def _memory_node(self, state: AgentState) -> AgentState:
        log("memory", "Retrieving memories...")
        query = state["user_input"]
        session_id = state["session_id"]
        retrieved = self.memory_store.retrieve(query, session_filter=session_id)
        state["memory_items"] = retrieved
        log("memory", f"Retrieved {len(retrieved)} memories")
        return state

    def _decision_node(self, state: AgentState) -> AgentState:
        log("decision", "Generating plan...")
        plan = self.decision_service.generate_plan(
            state["perception"],
            state["memory_items"],
            state["tool_descriptions"]
        )
        state["decision"] = plan
        if plan.startswith("FINAL_ANSWER:"):
            state["final_answer"] = plan
            state["should_continue"] = False
        log("decision", f"Plan: {plan}")
        return state

    async def _tool_node(self, state: AgentState) -> AgentState:
        log("tool", "Executing tool...")
        decision = state["decision"]
        if decision.startswith("FINAL_ANSWER:"):
            state["final_answer"] = decision
            state["should_continue"] = False
            return state

        # In Hexagonal, we rely on the ToolExecutor adapter to handle parsing if possible,
        # or we do it here. Our MCP Adapter implementation had `execute_raw` which parses string.
        # However, interface had `execute(name, args)`. 
        # Let's check if our adapter supports string parsing or if we need to do it.
        # For this refactor I added `execute_raw` to the adapter specifically for this use case.
        # But `ToolExecutor` interface didn't have it. Ideally we should have added it to interface.
        # Let's assume we cast or updated interface.
        
        try:
             # We rely on convention that our adapter has this method or we used a concrete class type hint
             # Strict hexagonal would require the interface to have `execute_from_string`
             if hasattr(self.tool_executor, 'execute_raw'):
                 result = await self.tool_executor.execute_raw(decision)
             else:
                 # Fallback/Mock logic
                 raise NotImplementedError("Executor does not support raw string execution")

             state["tool_result"] = result
             state["error"] = None
             log("tool", f"Result: {result.result}")
        except Exception as e:
            state["error"] = str(e)
            state["tool_result"] = None
            log("error", f"Tool failed: {e}")
        
        return state

    def _memory_update_node(self, state: AgentState) -> AgentState:
        tool_result = state.get("tool_result")
        if tool_result and not state.get("error"):
            record = MemoryRecord(
                text=f"Tool {tool_result.tool_name}: {tool_result.result}",
                type="tool_output",
                session_id=state["session_id"],
                user_query=state["user_input"],
                tags=[tool_result.tool_name],
                tool_name=tool_result.tool_name
            )
            self.memory_store.add(record)
            
            # Update input for next loop
            state["user_input"] = (
                f"Original: {state['original_query']}\n"
                f"Output: {tool_result.result}\n"
                f"Next?"
            )
            state["step"] += 1
        return state

    def _route_decision(self, state: AgentState) -> str:
        if state.get("final_answer") or state["decision"].startswith("FINAL_ANSWER:"):
            return "end"
        if state["decision"].startswith("FUNCTION_CALL:"):
            return "tool"
        return "end"

    def _check_continue(self, state: AgentState) -> str:
        if state.get("final_answer"): return "end"
        if state.get("error"): return "error"
        if state["step"] >= state["max_steps"]:
            state["final_answer"] = "FINAL_ANSWER: [Max steps]"
            return "end"
        return "continue"
    
    def _error_handler(self, state: AgentState) -> AgentState:
        state["final_answer"] = f"FINAL_ANSWER: Error: {state.get('error')}"
        state["should_continue"] = False
        return state

    def build(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("perception", self._perception_node)
        workflow.add_node("memory", self._memory_node)
        workflow.add_node("decision", self._decision_node)
        workflow.add_node("mcp_tool_execution", self._tool_node)
        workflow.add_node("memory_update", self._memory_update_node)
        workflow.add_node("error_handler", self._error_handler)

        workflow.set_entry_point("perception")

        workflow.add_edge("perception", "memory")
        workflow.add_edge("memory", "decision")
        
        workflow.add_conditional_edges(
            "decision",
            self._route_decision,
            {"tool": "mcp_tool_execution", "end": END}
        )
        
        workflow.add_edge("mcp_tool_execution", "memory_update")
        
        workflow.add_conditional_edges(
            "memory_update",
            self._check_continue,
            {"continue": "perception", "end": END, "error": "error_handler"}
        )
        
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
