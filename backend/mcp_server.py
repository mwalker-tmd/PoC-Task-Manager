from fastapi import FastAPI, HTTPException, Request
from fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from backend.graphs.task_agent import graph, TaskAgentState
from backend.types import TaskMetadata, SubtaskMetadata
from fastapi.responses import JSONResponse
from langgraph.errors import GraphInterrupt
from fastapi.exceptions import RequestValidationError

# A FastAPI app
app = FastAPI()

class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, description="The task to be processed")

class TaskResponse(BaseModel):
    task: str
    subtasks: Optional[List[str]] = None
    status: str
    message: Optional[str] = None
    needs_input: Optional[bool] = None
    prompt: Optional[str] = None

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert FastAPI request validation errors to 400 Bad Request."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Convert Pydantic validation errors to 400 Bad Request."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """
    Create a new task using the LangGraph task agent.
    This will:
    1. Extract and validate the task
    2. Optionally break it into subtasks
    3. Return the final task and subtasks
    
    If the graph needs user input, it will return a response with needs_input=True
    and a prompt message that should be shown to the user.
    """
    try:
        # Initialize the task agent state
        state = TaskAgentState(input=request.task)
        
        # Run the task agent graph
        result = await graph.ainvoke(state)
        
        # Extract task metadata from result
        task_metadata = result.get("task_metadata")
        if not task_metadata:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract task metadata"
            )
        
        # Return the successful result
        return TaskResponse(
            task=task_metadata.task,
            subtasks=result.get("subtask_metadata", {}).get("subtasks"),
            status="success",
            message="Task created successfully"
        )
        
    except GraphInterrupt as e:
        # This is not an error - the graph needs user input
        return TaskResponse(
            task=request.task,
            status="pending",
            needs_input=True,
            prompt=str(e),
            message="Additional information required"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Handle validation errors
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        # Log the error and return a 500
        print(f"Error processing task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing task"
        )

# Create an MCP server from your FastAPI app
mcp = FastMCP.from_fastapi(app=app)

if __name__ == "__main__":
    mcp.run()  # Start the MCP server