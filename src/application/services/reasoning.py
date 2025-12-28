from typing import List, Optional
from domain.llm.llm_port import LLMProvider
from domain.memory.memory_port import MemoryRecord
from src.domain.perception.models import PerceptionResult
from src.domain.decision.models import DecisionResult

class DecisionService:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def generate_plan(
        self,
        perception: PerceptionResult,
        memory_items: List[MemoryRecord],
        tool_descriptions: Optional[str] = None
    ) -> DecisionResult:
        
        memory_texts = "\n".join(f"- {m.text}" for m in memory_items) or "None"
        tool_context = f"\nYou have access to the following tools:\n{tool_descriptions}" if tool_descriptions else "No tools available."
        
        system_instructions = f"""
        You are an intelligent E-commerce Orchestrator. Your goal is to help the user by either calling a tool to get information or providing a final answer.
        
        Current State:
        - User Intent: {perception.intent}
        - Extracted Entities: {', '.join(perception.entities)}
        - User Query: "{perception.user_input}"
        
        History/Memory:
        {memory_texts}

        {tool_context}

        Guidelines:
        1. ANALYZE variables "User Intent" and "Memory". 
        2. If you need more information (like searching products, getting details), choose 'tool_call'.
        3. If you have enough information in Memory to answer the User Request, choose 'final_answer'.
        4. If you have search results, check if you need to refine/rank them using a tool.
        5. ALWAYS provide a 'thought' explaining your decision.
        """
        
        try:
            # We use the structure checking capabilities of the LLM adapter
            return self.llm.generate_structured(system_instructions, DecisionResult)
        except Exception as e:
            # Fallback for safety, though structured generation usually handles schema enforcement
            return DecisionResult(
                thought=f"Error during decision generation: {e}",
                decision_type="final_answer",
                final_answer="I encountered an internal error while deciding what to do."
            )

