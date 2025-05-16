from fastapi.testclient import TestClient
from backend.mcp_server import app

client = TestClient(app)

def test_list_items():
    response = client.get("/items")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_item():
    response = client.get("/items/42")
    assert response.status_code == 200
    assert response.json()["id"] == 42

def test_create_item():
    response = client.post("/items", params={"name": "Test Item"})
    assert response.status_code == 200
    assert response.json()["name"] == "Test Item"

def test_create_task():
    response = client.post("/tasks", params={"task": "Test Task"})
    assert response.status_code == 200
    assert response.json()["task"] == "Test Task" 