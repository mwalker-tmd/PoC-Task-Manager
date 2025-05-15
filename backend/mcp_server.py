from fastapi import FastAPI
from fastmcp import FastMCP


# A FastAPI app
app = FastAPI()

@app.get("/items")
def list_items():
    return [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]

@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"id": item_id, "name": f"Item {item_id}"}

@app.post("/items")
def create_item(name: str):
    return {"id": 3, "name": name}

@app.post("/tasks")
def create_task(task: str):
    
    return {"id": 3, "name": task}

# Create an MCP server from your FastAPI app
mcp = FastMCP.from_fastapi(app=app)

if __name__ == "__main__":
    mcp.run()  # Start the MCP server