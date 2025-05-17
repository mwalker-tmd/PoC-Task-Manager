from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from typing import Optional, List

from backend.tools.task_tools import (
    extract_task,
    analyze_subtasks,
    judge_task,
    judge_subtasks,
    save_task_to_db,
    ask_to_subtask,
    create_clarifying_questions,
    receive_clarification_feedback
)

class TaskAgentState(BaseModel):
    input: Optional[str] = None
    task: Optional[str] = None
    subtasks: Optional[List[str]] = None
    clarification_needed: Optional[List[str]] = None
    clarification_questions: Optional[List[str]] = None
    confirmed: Optional[bool] = False
    judgment: Optional[str] = None
    user_feedback: Optional[str] = None
    subtask_decision: Optional[str] = None

# Node Definitions
def extract_task_node(state):
    result = extract_task(state.input)
    return {"task": result["task"]}

def judge_task_node(state):
    result = judge_task(state.input, state.task)
    return {"judgment": result["status"]}

def judge_subtasks_node(state):
    result = judge_subtasks(state.input, state.task, state.subtasks or [])
    return {"judgment": result["status"]}

def ask_to_subtask_node(state):
    result = ask_to_subtask(state.task)
    return {"subtask_decision": result["decision"]}

def create_clarifying_questions_node(state):
    result = create_clarifying_questions(state.task)
    return {"clarification_questions": result["questions"]}

def ask_clarifying_questions_node(state):
    return {}  # handled externally / human-in-the-loop

def receive_clarification_feedback_node(state):
    result = receive_clarification_feedback(state.clarification_questions)
    return {"input": result["updated_input"]}

def generate_subtasks_node(state):
    result = analyze_subtasks(state.task)
    return {
        "subtasks": result["subtasks"],
        "clarification_needed": result["missing_info"]
    }

def create_task_node(state):
    result = save_task_to_db(state.task, state.subtasks)
    return {"confirmed": True}

# Graph Construction
builder = StateGraph(TaskAgentState)

builder.add_node("extract_task", RunnableLambda(extract_task_node))
builder.add_node("judge_task", RunnableLambda(judge_task_node))
builder.add_node("judge_subtasks", RunnableLambda(judge_subtasks_node))
builder.add_node("ask_to_subtask", RunnableLambda(ask_to_subtask_node))
builder.add_node("create_clarifying_questions", RunnableLambda(create_clarifying_questions_node))
builder.add_node("ask_clarifying_questions", RunnableLambda(ask_clarifying_questions_node))
builder.add_node("receive_clarification_feedback", RunnableLambda(receive_clarification_feedback_node))
builder.add_node("generate_subtasks", RunnableLambda(generate_subtasks_node))
builder.add_node("create_task", RunnableLambda(create_task_node))

# Transitions
builder.set_entry_point("extract_task")
builder.add_edge("extract_task", "judge_task")
builder.add_conditional_edges("judge_task", lambda s: s.judgment, {
    "approved": "ask_to_subtask",
    "needs_revision": "create_clarifying_questions"
})
builder.add_edge("create_clarifying_questions", "ask_clarifying_questions")
builder.add_edge("ask_clarifying_questions", "receive_clarification_feedback")
# After clarification, decide where to go based on whether subtasks exist
builder.add_conditional_edges(
    "receive_clarification_feedback",
    lambda s: "generate_subtasks" if s.subtasks else "extract_task",
    {
        "generate_subtasks": "generate_subtasks",
        "extract_task": "extract_task"
    }
)
builder.add_conditional_edges("ask_to_subtask", lambda s: s.subtask_decision, {
    "no": "create_task",
    "yes": "generate_subtasks"
})
builder.add_edge("generate_subtasks", "judge_subtasks")
builder.add_conditional_edges("judge_subtasks", lambda s: s.judgment, {
    "approved": "create_task",
    "needs_revision": "create_clarifying_questions"
})
builder.add_edge("create_task", END)

# Compile
graph = builder.compile()