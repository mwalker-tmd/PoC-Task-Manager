from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from typing import Optional, List
from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from backend.types import TaskMetadata, TaskJudgment, JudgmentType, SubtaskDecision, TaskAgentState
from backend.tools import (
    extract_task,
    judge_task,
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
    """
    Pause execution and ask the user if they want help breaking the task into subtasks.
    Retries up to 2 times before defaulting to "no".
    """
    if not state.subtask_decision:
        state.subtask_decision = SubtaskDecision(value=None, retries=0)
        prompt_message = "Would you like help breaking this task into subtasks? (yes/no)"
    else:
        prompt_message = ("Sorry, I was unable to determine if that was a yes or a no.\n\n" 
            "Would you like help breaking this task into subtasks? (yes/no)"
        )

    if not state.subtask_decision.value:
        state.subtask_decision.retries += 1

        if state.subtask_decision.retries >= 3:
            state.subtask_decision.value = "no"
        else:
            raise GraphInterrupt(
                Interrupt(
                    value=prompt_message,
                    resumable=True
                )
            )
    
    # Clear user feedback after processing
    state.user_feedback = ""
    return state

def generate_subtasks_node(state: TaskAgentState) -> TaskAgentState:
    result = generate_subtasks(state.task_metadata)
    state.subtask_metadata = result
    return state

def judge_subtasks_node(state: TaskAgentState) -> TaskAgentState:
    result = judge_subtasks(state.task_metadata, state.subtask_metadata)
    state.subtask_judgment = result
    return state

def create_clarifying_questions_node(state: TaskAgentState) -> TaskAgentState:
    result = create_clarifying_questions(state.subtask_metadata.missing_info)
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
builder.add_conditional_edges("judge_task", lambda s: s.task_judgment.judgment.value, {
    JudgmentType.PASS.value: "ask_to_subtask",
    JudgmentType.FAIL.value: "create_clarifying_questions"
})
builder.add_edge("create_clarifying_questions", "ask_clarifying_questions")
builder.add_edge("ask_clarifying_questions", "receive_clarification_feedback")
builder.add_edge("receive_clarification_feedback", "extract_task")
builder.add_conditional_edges("ask_to_subtask", lambda s: s.subtask_decision, {
    "yes": "generate_subtasks",
    "no": "create_task"
})
builder.add_edge("generate_subtasks", "judge_subtasks")
builder.add_conditional_edges("judge_subtasks", lambda s: s.subtask_judgment.judgment.value, {
    JudgmentType.PASS.value: "create_task",
    JudgmentType.FAIL.value: "create_clarifying_questions"
})
builder.add_edge("create_task", END)

# Compile the graph
graph = builder.compile()
