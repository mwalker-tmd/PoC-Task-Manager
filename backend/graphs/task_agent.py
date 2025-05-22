from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from typing import Optional, List

from backend.types import TaskMetadata, TaskJudgment, TaskAgentState
from backend.tools import (
    extract_task,
    judge_task,
    ask_to_subtask,
    generate_subtasks,
    judge_subtasks,
    create_clarifying_questions,
    ask_clarifying_questions,
    receive_clarification_feedback,
    create_task
)

# Node definitions

def extract_task_node(state: TaskAgentState) -> TaskAgentState:
    result = extract_task(state)
    state.task_metadata = result
    return state

def judge_task_node(state: TaskAgentState) -> TaskAgentState:
    result = judge_task(state.task_metadata)
    state.task_judgment = result
    return state

def ask_to_subtask_node(state: TaskAgentState) -> TaskAgentState:
    decision = ask_to_subtask(state.task_metadata.task)
    state.subtask_decision = decision.get("decision")
    return state

def generate_subtasks_node(state: TaskAgentState) -> TaskAgentState:
    result = generate_subtasks(state.task_metadata.task)
    state.subtasks = result.get("subtasks")
    state.missing_info = result.get("missing_info")
    return state

def judge_subtasks_node(state: TaskAgentState) -> TaskAgentState:
    result = judge_subtasks(state.task_metadata.task, state.subtasks)
    state.subtask_judgment = result.get("judgment")
    return state

def create_clarifying_questions_node(state: TaskAgentState) -> TaskAgentState:
    result = create_clarifying_questions(state.missing_info)
    state.clarification_questions = result.get("questions")
    return state

def ask_clarifying_questions_node(state: TaskAgentState) -> TaskAgentState:
    return state  # UI interaction stub

def receive_clarification_feedback_node(state: TaskAgentState) -> TaskAgentState:
    result = receive_clarification_feedback(state.clarification_questions)
    state.input = result.get("updated_input")
    return state

def create_task_node(state: TaskAgentState) -> TaskAgentState:
    create_task(state.task_metadata.task, state.subtasks)
    state.confirmed = True
    return state

# Build the graph
builder = StateGraph(TaskAgentState)

builder.add_node("extract_task", RunnableLambda(extract_task_node))
builder.add_node("judge_task", RunnableLambda(judge_task_node))
builder.add_node("ask_to_subtask", RunnableLambda(ask_to_subtask_node))
builder.add_node("generate_subtasks", RunnableLambda(generate_subtasks_node))
builder.add_node("judge_subtasks", RunnableLambda(judge_subtasks_node))
builder.add_node("create_clarifying_questions", RunnableLambda(create_clarifying_questions_node))
builder.add_node("ask_clarifying_questions", RunnableLambda(ask_clarifying_questions_node))
builder.add_node("receive_clarification_feedback", RunnableLambda(receive_clarification_feedback_node))
builder.add_node("create_task", RunnableLambda(create_task_node))

builder.set_entry_point("extract_task")

builder.add_edge("extract_task", "judge_task")
builder.add_conditional_edges("judge_task", lambda s: s.task_judgment, {
    "pass": "ask_to_subtask",
    "fail": "create_clarifying_questions"
})
builder.add_edge("create_clarifying_questions", "ask_clarifying_questions")
builder.add_edge("ask_clarifying_questions", "receive_clarification_feedback")
builder.add_edge("receive_clarification_feedback", "extract_task")
builder.add_conditional_edges("ask_to_subtask", lambda s: s.subtask_decision, {
    "yes": "generate_subtasks",
    "no": "create_task"
})
builder.add_edge("generate_subtasks", "judge_subtasks")
builder.add_conditional_edges("judge_subtasks", lambda s: s.subtask_judgment, {
    "pass": "create_task",
    "fail": "create_clarifying_questions"
})
builder.add_edge("create_task", END)

# Compile the graph
graph = builder.compile()
