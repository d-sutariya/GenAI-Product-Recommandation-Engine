
import pytest
from unittest.mock import Mock, patch, MagicMock
from client.infrastructure.llm.huggingface_adapter import HFLLMAdapter
from pydantic import BaseModel

class TestSchema(BaseModel):
    field: str

@pytest.fixture
def mock_inference_client():
    with patch('client.infrastructure.llm.huggingface_adapter.InferenceClient') as mock:
        yield mock

@pytest.fixture
def mock_env():
    with patch('os.getenv') as mock_env:
        mock_env.return_value = "fake_token"
        yield mock_env

class TestHFLLMAdapter:
    def test_init(self, mock_env, mock_inference_client):
        """Test initialization."""
        adapter = HFLLMAdapter(repo_id="test/repo")
        assert adapter.repo_id == "test/repo"
        mock_inference_client.assert_called_once()

    def test_init_no_token(self):
        """Test initialization failure without token."""
        with patch('os.getenv', return_value=None):
            with pytest.raises(ValueError):
                HFLLMAdapter()

    def test_generate_structured(self, mock_env, mock_inference_client):
        """Test structured generation."""
        adapter = HFLLMAdapter()
        
        # Mock chat completion response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"field": "test_value"}'
        adapter.client.chat_completion.return_value = mock_response
        
        result = adapter.generate_structured("prompt", TestSchema)
        
        assert isinstance(result, TestSchema)
        assert result.field == "test_value"

    def test_generate(self, mock_env, mock_inference_client):
        """Test text generation."""
        adapter = HFLLMAdapter()
        
        # Mock chat completion response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "generated text"
        adapter.client.chat_completion.return_value = mock_response
        
        result = adapter.generate("prompt")
        
        assert result == "generated text"
        adapter.client.chat_completion.assert_called()
