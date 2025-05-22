import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_openai():
    with patch('backend.tools.task_tools.get_client') as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create = Mock()
        yield mock_client 