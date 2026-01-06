
import pytest
from unittest.mock import Mock, MagicMock
from src.application.services.perception import PerceptionService
from src.domain.perception.models import PerceptionResult

@pytest.fixture
def mock_llm():
    return Mock()

class TestPerceptionService:
    def test_analyze_input_success(self, mock_llm):
        """Test successful perception analysis."""
        service = PerceptionService(mock_llm)
        expected_result = PerceptionResult(
            user_input="test query",
            intent="search",
            entities=["item"],
            tool_hint="search_tool"
        )
        mock_llm.generate_structured.return_value = expected_result
        
        result = service.analyze_input("test query")
        
        assert result == expected_result
        mock_llm.generate_structured.assert_called_once()

    def test_analyze_input_failure_fallback(self, mock_llm):
        """Test fallback when LLM fails."""
        service = PerceptionService(mock_llm)
        mock_llm.generate_structured.side_effect = Exception("LLM Error")
        
        result = service.analyze_input("test query")
        
        assert result.user_input == "test query"
        assert result.intent == "product_search"
        # Should return fallback defaults
