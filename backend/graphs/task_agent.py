from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from typing import Optional, List
from backend.tools.task_tools import extract_task, analyze_subtasks, clarify, review

# Define the input/output state as a Pydantic model
class TaskAgentState(BaseModel):
    input: Optional[str] = None
    task: Optional[str] = None
    subtasks: Optional[List[str]] = None
    clarification_needed: Optional[List[str]] = None

# Define tool-wrapped steps
def extract_task_node(state):
    result = extract_task(state.input)
    return {"task": result["task"]}

def analyze_subtasks_node(state):
    result = analyze_subtasks(state.task)
    return {
        "subtasks": result["subtasks"],
        "clarification_needed": result["missing_info"]
    }

# Set up the graph
builder = StateGraph(TaskAgentState)

builder.add_node("extract_task", RunnableLambda(extract_task_node))
builder.add_node("analyze_subtasks", RunnableLambda(analyze_subtasks_node))

builder.set_entry_point("extract_task")
builder.add_edge("extract_task", "analyze_subtasks")
builder.add_edge("analyze_subtasks", END)

graph = builder.compile()
