from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel
from typing import Optional, List
from backend.graphs.task_agent import graph, TaskAgentState
from backend.types import TaskMetadata, SubtaskMetadata

# A FastAPI app
app = FastAPI()

class TaskRequest(BaseModel):
    task: str

class TaskResponse(BaseModel):
    task: str
    subtasks: Optional[List[str]] = None
    status: str
    message: Optional[str] = None

@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """
    Create a new task using the LangGraph task agent.
    This will:
    1. Extract and validate the task
    2. Optionally break it into subtasks
    3. Return the final task and subtasks
    """
    try:
        # Initialize the task agent state
        state = TaskAgentState(input=request.task)
        
        # Run the task agent graph
        result = await graph.ainvoke(state)
        
        # Check if the task was confirmed
        if not result.confirmed:
            raise HTTPException(
                status_code=400,
                detail="Task was not confirmed by the agent"
            )
        
        # Return the successful result
        return TaskResponse(
            task=result.task_metadata.task,
            subtasks=result.subtask_metadata.subtasks if result.subtask_metadata else None,
            status="success",
            message="Task created successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing task: {str(e)}"
        )

# Create an MCP server from your FastAPI app
mcp = FastMCP.from_fastapi(app=app)

if __name__ == "__main__":
    mcp.run()  # Start the MCP server