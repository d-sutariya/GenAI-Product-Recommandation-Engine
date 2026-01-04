
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from src.application.services.agent_orchestrator import AgentWorkflow
from src.application.services.perception import PerceptionService
from src.application.services.reasoning import DecisionService
from src.domain.perception.models import PerceptionResult
from src.domain.decision.models import DecisionResult
from src.domain.memory.memory_port import MemoryRecord
from src.domain.memory.memory_port import MemoryStore
from src.domain.tools.models import ToolCallResult

class MockMemoryStore(MemoryStore):
    def __init__(self):
        self.memories = []
    
    def add(self, record: MemoryRecord):
        self.memories.append(record)
        
    def retrieve(self, query, session_filter=None, top_k=5):
        return self.memories

@pytest.fixture
def mock_dependencies():
    llm = Mock()
    memory_store = MockMemoryStore()
    tool_executor = AsyncMock()
    
    return {
        "llm": llm,
        "memory_store": memory_store,
        "tool_executor": tool_executor
    }

@pytest.fixture
def workflow_services(mock_dependencies):
    perception_service = PerceptionService(mock_dependencies["llm"])
    decision_service = DecisionService(mock_dependencies["llm"])
    
    workflow = AgentWorkflow(
        perception_service=perception_service,
        decision_service=decision_service,
        memory_store=mock_dependencies["memory_store"],
        tool_executor=mock_dependencies["tool_executor"]
    )
    
    return workflow.build()

@pytest.mark.asyncio
async def test_workflow_end_to_end_mocked_flow(workflow_services, mock_dependencies):
    """
    Tests the full flow: 
    1. Perception (Mock LLM)
    2. Decision -> Tool Call (Mock LLM)
    3. Tool Execution (Mock ToolExecutor)
    4. Memory Update
    5. Loop -> Decision -> Final Answer (Mock LLM)
    """
    llm = mock_dependencies["llm"]
    tool_executor = mock_dependencies["tool_executor"]
    memory_store = mock_dependencies["memory_store"]

    # 1. Setup Mock Responses for the Sequence
    
    # Perception Response
    perception_result = PerceptionResult(
        user_input="find green shoes",
        intent="search",
        entities=["green", "shoes"],
        tool_hint="search"
    )

    # Decision 1: Call Tool
    decision_tool = DecisionResult(
        thought="searching for shoes",
        decision_type="tool_call",
        tool_name="search_products",
        tool_input={"query": "green shoes"}
    )
    
    # Decision 2: Final Answer
    decision_final = DecisionResult(
        thought="found them",
        decision_type="final_answer",
        final_answer="Here are your green shoes."
    )

    # Configure side_effects for LLM calls    
    def generate_side_effect(prompt, schema):
        if schema == PerceptionResult:
            return perception_result
        if schema == DecisionResult:
            # If the prompt contains the result from our tool execution, we know it's the second pass
            if "Green Shoe A" in str(prompt): 
                 return decision_final
            return decision_tool
        return None
        
    llm.generate_structured.side_effect = generate_side_effect

    # Tool Execution Result
    tool_executor.execute.return_value = ToolCallResult(
        tool_name="search_products",
        arguments={"query": "green shoes"},
        result="[Green Shoe A, Green Shoe B]"
    )

    # 2. Execute Workflow
    initial_state = {
        "user_input": "find green shoes",
        "original_query": "find green shoes",
        "session_id": "test-session",
        "perception": None,
        "memory_items": [],
        "decision": None,
        "tool_result": None,
        "mcp_session": None,
        "mcp_tools": [],
        "tool_descriptions": "search_products: search for stuff",
        "step": 0,
        "max_steps": 5,
        "final_answer": None,
        "error": None,
        "should_continue": True
    }

    final_state = await workflow_services.ainvoke(initial_state)

    # 3. Assertions
    
    # Check Final Answer
    assert final_state["final_answer"] == "Here are your green shoes."
    
    # Check Tool Execution
    tool_executor.execute.assert_called_once()
    assert tool_executor.execute.call_args[0][0] == "search_products"
    
    # Check Memory Update (MockMemoryStore should have 1 item)
    assert len(memory_store.memories) == 1
    assert "Green Shoe A" in memory_store.memories[0].text

    # Verify Logic Flow
    # Should have called LLM for perception (at least once) and decision (2 times)
    assert llm.generate_structured.call_count >= 3 
