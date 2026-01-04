
import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.infrastructure.tools.mcp_tool_adapter import MCPToolAdapter

@pytest.fixture
def mock_session():
    session = AsyncMock()
    return session

@pytest.mark.asyncio
class TestMCPToolAdapter:
    async def test_list_tools(self, mock_session):
        """Test listing tools."""
        mock_tool = Mock()
        mock_tool.name = "tool1"
        mock_tool.description = "desc1"
        mock_session.list_tools.return_value.tools = [mock_tool]
        
        adapter = MCPToolAdapter(mock_session)
        tools = await adapter.list_tools()
        
        assert len(tools) == 1
        assert tools[0].name == "tool1"
        assert adapter.cached_tools is not None

    async def test_get_tool_descriptions(self, mock_session):
        """Test getting tool descriptions."""
        mock_tool = Mock()
        mock_tool.name = "tool1"
        mock_tool.description = "desc1"
        mock_session.list_tools.return_value.tools = [mock_tool]
        
        adapter = MCPToolAdapter(mock_session)
        await adapter.list_tools() # Populate cache
        
        desc = adapter.get_tool_descriptions()
        assert "tool1: desc1" in desc

    async def test_execute_string_result(self, mock_session):
        """Test executing tool with string result."""
        mock_result = Mock()
        mock_result.content = "result_text"
        mock_session.call_tool.return_value = mock_result
        
        adapter = MCPToolAdapter(mock_session)
        result = await adapter.execute("tool1", {"arg": 1})
        
        assert result.tool_name == "tool1"
        assert result.result == "result_text"

    async def test_execute_list_result(self, mock_session):
        """Test executing tool with list result."""
        content_item = Mock()
        content_item.text = "item1"
        mock_result = Mock()
        mock_result.content = [content_item]
        mock_session.call_tool.return_value = mock_result
        
        adapter = MCPToolAdapter(mock_session)
        result = await adapter.execute("tool1", {})
        
        assert result.result == ["item1"]

