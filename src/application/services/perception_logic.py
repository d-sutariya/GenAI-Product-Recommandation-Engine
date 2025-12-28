from src.domain.ports.output.llm_port import LLMProvider
from src.domain.perception.models import PerceptionResult

class PerceptionService:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def analyze_input(self, user_input: str) -> PerceptionResult:
        """Extracts intent, entities, and tool hints using LLM"""
        prompt = f"""
            You are an E-commerce Product Search Agent. Your task is to extract structured information from a user's product-related query.

            Input: "{user_input}"
        """
        try:
            parsed = self.llm.generate_structured(prompt, schema=PerceptionResult)
            return parsed
        except Exception as e:
            # Fallback
            return PerceptionResult(user_input=user_input)
