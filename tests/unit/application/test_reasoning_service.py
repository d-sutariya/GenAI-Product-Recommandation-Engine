
import pytest
from unittest.mock import Mock, MagicMock
from client.application.services.reasoning import DecisionService
from client.domain.perception.models import PerceptionResult
from client.domain.decision.models import DecisionResult
from client.domain.memory.models import MemoryRecord

@pytest.fixture
def mock_llm():
    return Mock()

class TestDecisionService:
    def test_generate_plan_success(self, mock_llm):
        """Test successful plan generation."""
        service = DecisionService(mock_llm)
        perception = PerceptionResult(user_input="test")
        memories = [MemoryRecord(text="mem1")]
        
        expected_decision = DecisionResult(
            thought="thought",
            decision_type="final_answer",
            final_answer="answer"
        )
        mock_llm.generate_structured.return_value = expected_decision
        
        result = service.generate_plan(perception, memories)
        
        assert result == expected_decision
        mock_llm.generate_structured.assert_called_once()
        
        # Verify prompt construction (partially)
        call_args = mock_llm.generate_structured.call_args
        prompt = call_args[0][0]
        assert "User Query: \"test\"" in prompt
        assert "mem1" in prompt

    def test_generate_plan_failure_fallback(self, mock_llm):
        """Test fallback when LLM fails."""
        service = DecisionService(mock_llm)
        perception = PerceptionResult(user_input="test")
        
        mock_llm.generate_structured.side_effect = Exception("LLM Error")
        
        result = service.generate_plan(perception, [])
        
        assert result.decision_type == "final_answer"
        assert "error" in result.thought.lower()
