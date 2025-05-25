from fastapi.testclient import TestClient
from backend.mcp_server import app
import pytest

client = TestClient(app)

def test_create_task():
    """Test creating a task through the API."""
    response = client.post(
        "/tasks",
        json={"task": "Test Task"}  # Use json parameter instead of params
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "Test Task"
    assert data["status"] == "success"
    assert data["message"] == "Task created successfully"

def test_create_task_invalid_input():
    """Test creating a task with invalid input."""
    response = client.post(
        "/tasks",
        json={"invalid": "field"}  # Missing required 'task' field
    )
    assert response.status_code == 400  # Bad request
    assert "detail" in response.json()

def test_create_task_empty_task():
    """Test creating a task with an empty task string."""
    response = client.post(
        "/tasks",
        json={"task": ""}  # Empty task string
    )
    assert response.status_code == 400  # Bad request 