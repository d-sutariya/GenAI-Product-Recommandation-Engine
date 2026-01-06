
import pytest
from pydantic import ValidationError
from src.domain.perception.models import PerceptionResult
from src.domain.decision.models import DecisionResult
from src.domain.memory.models import MemoryRecord
from src.domain.tools.models import ToolCallResult

class TestPerceptionResult:
    def test_perception_result_defaults(self):
        """Test default values for PerceptionResult."""
        perception = PerceptionResult(user_input="test query")
        assert perception.user_input == "test query"
        assert perception.modified_user_input is None
        assert perception.intent is None
        assert perception.entities == []
        assert perception.tool_hint is None

    def test_perception_result_validation(self):
        """Test basic validation."""
        with pytest.raises(ValidationError):
            PerceptionResult()  # user_input is required

    def test_perception_result_full(self):
        """Test with all fields populated."""
        perception = PerceptionResult(
            user_input="original",
            modified_user_input="modified",
            intent="search",
            entities=["a", "b"],
            tool_hint="tool_a"
        )
        assert perception.entities == ["a", "b"]
        assert perception.intent == "search"


class TestDecisionResult:
    def test_decision_result_tool_call(self):
        """Test decision result for tool call."""
        decision = DecisionResult(
            thought="thinking",
            decision_type="tool_call",
            tool_name="search",
            tool_input={"q": "foo"}
        )
        assert decision.decision_type == "tool_call"
        assert decision.tool_name == "search"

    def test_decision_result_final_answer(self):
        """Test decision result for final answer."""
        decision = DecisionResult(
            thought="done",
            decision_type="final_answer",
            final_answer="The answer"
        )
        assert decision.decision_type == "final_answer"
        assert decision.final_answer == "The answer"

    def test_decision_result_validation(self):
        """Test valid decision types."""
        with pytest.raises(ValidationError):
            DecisionResult(
                thought="thinking",
                decision_type="invalid_type"
            )


class TestMemoryRecord:
    def test_memory_record_defaults(self):
        """Test default values."""
        record = MemoryRecord(text="remember this")
        assert record.text == "remember this"
        assert record.type == "fact"
        assert record.tags == []

    def test_memory_record_full(self):
        """Test all fields."""
        record = MemoryRecord(
            text="text",
            type="tool_output",
            timestamp="2023-01-01",
            tool_name="search",
            user_query="query",
            tags=["tag1"],
            session_id="sess1"
        )
        assert record.session_id == "sess1"
        assert "tag1" in record.tags


class TestToolCallResult:
    def test_tool_call_result_structure(self):
        """Test tool call result structure."""
        result = ToolCallResult(
            tool_name="test_tool",
            arguments={"arg": 1},
            result={"status": "ok"}
        )
        assert result.tool_name == "test_tool"
        assert result.result["status"] == "ok"
