import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from client.infrastructure.memory.faiss_memory_adapter import FaissMemoryAdapter
from client.domain.memory.models import MemoryRecord

@pytest.fixture
def mock_genai_client():
    with patch('client.infrastructure.memory.faiss_memory_adapter.genai.Client') as mock:
        client_instance = mock.return_value
        # Mock embeddings response
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768
        
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        
        client_instance.models.embed_content.return_value = mock_response
        yield client_instance

@pytest.fixture
def mock_faiss():
    with patch('client.infrastructure.memory.faiss_memory_adapter.faiss') as mock:
        index_mock = Mock()
        mock.IndexFlatL2.return_value = index_mock
        # Mock search return: distances (D) and indices (I)
        index_mock.search.return_value = (np.array([[0.0]]), np.array([[0]]))
        yield mock

class TestFaissMemoryAdapter:
    def test_init(self, mock_genai_client):
        """Test initialization."""
        adapter = FaissMemoryAdapter()
        assert adapter.data == []
        assert adapter.embeddings == []
        assert adapter.index is None
        mock_genai_client.models.embed_content.assert_not_called()

    def test_add_item(self, mock_genai_client, mock_faiss):
        """Test adding an item."""
        adapter = FaissMemoryAdapter()
        record = MemoryRecord(text="test memory")
        
        adapter.add(record)
        
        assert len(adapter.data) == 1
        assert len(adapter.embeddings) == 1
        # Check if FAISS index was created and added to
        mock_faiss.IndexFlatL2.assert_called_once()
        adapter.index.add.assert_called_once()

    def test_retrieve(self, mock_genai_client, mock_faiss):
        """Test retrieval."""
        adapter = FaissMemoryAdapter()
        record = MemoryRecord(text="test memory", session_id="sess1")
        adapter.add(record)
        
        # Setup search result to point to index 0
        adapter.index.search.return_value = (np.array([[0.1]]), np.array([[0]]))
        
        results = adapter.retrieve("query", top_k=1)
        
        assert len(results) == 1
        assert results[0].text == "test memory"
        adapter.index.search.assert_called()

    def test_retrieve_empty(self, mock_genai_client):
        """Test retrieve from empty store."""
        adapter = FaissMemoryAdapter()
        results = adapter.retrieve("query")
        assert results == []

    def test_retrieve_session_filter(self, mock_genai_client, mock_faiss):
        """Test retrieval with session filter."""
        adapter = FaissMemoryAdapter()
        record1 = MemoryRecord(text="mem1", session_id="sess1")
        record2 = MemoryRecord(text="mem2", session_id="sess2")
        adapter.add(record1)
        adapter.add(record2)
        
        # Stub search to return both
        adapter.index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        
        results = adapter.retrieve("query", session_filter="sess1")
        
        assert len(results) == 1
        assert results[0].session_id == "sess1"
