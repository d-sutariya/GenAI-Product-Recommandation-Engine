from typing import List, Optional
import time
from client.domain.memory.memory_port import MemoryStore
from client.domain.llm.llm_port import LLMProvider
from client.domain.memory.models import MemoryRecord
from client.utils.logger import log

class ClientHistoryRAGService:
    def __init__(self, memory_store: MemoryStore, llm_provider: LLMProvider):
        self.memory_store = memory_store
        self.llm_provider = llm_provider

    def add_interaction(self, user_id: str, client_message: str, agent_message: str, session_id: Optional[str] = None):
        """
        Stores the interaction in FAISS with user_id namespace.
        If the content length exceeds 10,000 characters, it is summarized before storage.
        """
        full_text = f"User: {client_message}\nAI: {agent_message}"
        content_to_store = full_text
        original_length = len(full_text)
        
        if original_length > 10000:
            log("rag_service", f"Interaction length {original_length} > 10000 for user {user_id}. Summarizing...")
            try:
                summary_prompt = f"Summarize the following conversation interaction efficiently, preserving key details and facts:\n\n{full_text}"
                summary = self.llm_provider.generate(summary_prompt)
                content_to_store = f"SUMMARY: {summary}"
                log("rag_service", "Summarization complete.")
            except Exception as e:
                log("rag_service", f"Summarization failed: {e}. Storing original text (truncated possibly by embedding limit).")
                pass
        
        record = MemoryRecord(
            text=content_to_store,
            type="conversation_history",
            timestamp=str(time.time()),
            user_id=user_id,
            session_id=session_id
        )
        
        self.memory_store.add(record)
        log("rag_service", f"Stored interaction for user {user_id}")

    def get_context(self, user_id: str, query: str, top_k: int = 2) -> str:
        """
        Retrieves relevant history for a user and query to be used as context for AI.
        """
        records = self.memory_store.retrieve(query, top_k=top_k, user_id=user_id)
        if not records:
            return ""
            
        context_str = "Relevant Past History:\n" + "\n".join([f"- {r.text}" for r in records])
        return context_str
