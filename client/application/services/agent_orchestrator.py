from langgraph.graph import StateGraph, END
from client.domain.shared.state import AgentState
from client.application.services.perception import PerceptionService
from client.application.services.reasoning import DecisionService
from client.domain.memory.memory_port import MemoryStore, MemoryRecord
from client.domain.memory.memory_port import MemoryStore, MemoryRecord
from client.domain.tools.tool_port import ToolExecutor
from client.utils.logger import log
import asyncio

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
        user_id = state.get("user_id")
        retrieved = self.memory_store.retrieve(query, session_filter=session_id, user_id=user_id)
        state["memory_items"] = retrieved
        log("memory", f"Retrieved {len(retrieved)} memories")
        return state

    def _decision_node(self, state: AgentState) -> AgentState:
        log("decision", "Generating plan...")
        decision = self.decision_service.generate_plan(
            state["perception"],
            state["memory_items"],
            state["tool_descriptions"]
        )
        state["decision"] = decision
        if decision.decision_type == "final_answer":
            state["final_answer"] = decision.final_answer
            # Don't end immediately if we have a recommendation; let add_to_cart handle it
            if decision.recommended_product:
                state["should_continue"] = True
            else:
                state["should_continue"] = False
        log("decision", f"Plan: {decision}")
        return state

    async def _tool_node(self, state: AgentState) -> AgentState:
        log("tool", "Executing tool...")
        decision = state["decision"]
        if decision.decision_type == "final_answer":
            state["final_answer"] = decision.final_answer
            state["should_continue"] = False

        elif decision.decision_type == "tool_call":
            tool_name = decision.tool_name
            tool_args = decision.tool_input
            try:
                result = await self.tool_executor.execute(tool_name, tool_args)
                state["tool_result"] = result
                log("tool", f"Tool {tool_name} executed successfully.")
            except Exception as e:
                log("tool", f"Tool execution failed: {e}")
                state["error"] = str(e)

        return state

    def _add_to_cart_node(self, state: AgentState) -> AgentState:
        log("cart", "Checking for product addition...")
        decision = state["decision"]
        product_name = decision.recommended_product
        
        if product_name:
            print(f"\n[Agent] I found a product: {product_name}.")
            print(f"[Agent] Do you want to add {product_name} to your basket? (yes/no)")
            
            user_response = input("User: ").strip().lower()
            
            if user_response in ["yes", "y", "sure", "ok", "add it"]:
                print(f"\n[System] {product_name} is added into the basket.\n")
                state["final_answer"] += f"\n\n(System: '{product_name}' was added to the basket.)"
            else:
                print(f"\n[System] Product not added.\n")
        
        state["should_continue"] = False
        return state

    def _memory_update_node(self, state: AgentState) -> AgentState:
        tool_result = state.get("tool_result")
        if tool_result and not state.get("error"):
            record = MemoryRecord(
                text=f"Tool {tool_result.tool_name}: {tool_result.result}",
                type="tool_output",
                session_id=state["session_id"],
                user_id=state.get("user_id"),
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
        decision = state["decision"]
        
        if decision.decision_type == "final_answer":
            if decision.recommended_product:
                return "add_to_cart"
            return "end"
        if state["decision"].decision_type == "tool_call":
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

        workflow.add_node("add_to_cart", self._add_to_cart_node)

        workflow.set_entry_point("perception")

        workflow.add_edge("perception", "memory")
        workflow.add_edge("memory", "decision")
        
        workflow.add_conditional_edges(
            "decision",
            self._route_decision,
            {
                "tool": "mcp_tool_execution", 
                "end": END,
                "add_to_cart": "add_to_cart"
            }
        )
        
        workflow.add_edge("mcp_tool_execution", "memory_update")
        workflow.add_edge("add_to_cart", END)
        
        workflow.add_conditional_edges(
            "memory_update",
            self._check_continue,
            {"continue": "perception", "end": END, "error": "error_handler"}
        )
        
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
