from client.domain.llm.llm_port import LLMProvider
from client.domain.perception.models import PerceptionResult

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
            # Fallback with basic defaults
            return PerceptionResult(
                user_input=user_input,
                modified_user_input=user_input,
                intent="product_search",
                entities=[],
                tool_hint="search_product_documents"
            )
