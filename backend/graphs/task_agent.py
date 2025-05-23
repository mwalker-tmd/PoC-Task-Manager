from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from typing import Optional, List
from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from backend.types import TaskMetadata, TaskJudgment, JudgmentType, SubtaskDecision, TaskAgentState, JudgmentRetry, SubtaskJudgment
from backend.tools import (
    extract_task,
    judge_task,
    generate_subtasks,
    judge_subtasks,
    create_task,
    retry_task_with_feedback,
    retry_subtasks_with_feedback,
    generate_task_clarification_prompt
)

# Node definitions
def extract_task_node(state: TaskAgentState) -> TaskAgentState:
    result = extract_task(state)
    state.task_metadata = result
    return state

def judge_task_node(state: TaskAgentState) -> TaskAgentState:
    """Judge the task and track retry attempts."""
    if state.task_judgment_retry is None:
        state.task_judgment_retry = JudgmentRetry()
    
    result = judge_task(state.task_metadata)
    state.task_judgment = result
    
    # Reset retry counter on pass
    if result.judgment == JudgmentType.PASS:
        state.task_judgment_retry.retries = 0
    else:
        state.task_judgment_retry.retries += 1
        
        # If we've hit max retries, force a pass
        if state.task_judgment_retry.retries >= state.task_judgment_retry.max_retries:
            state.task_judgment = TaskJudgment(
                judgment=JudgmentType.PASS,
                reason="Max retries reached, proceeding with task"
            )
    
    return state

def ask_to_subtask_node(state: TaskAgentState) -> TaskAgentState:
    """
    Pause execution and ask the user if they want help breaking the task into subtasks.
    Retries up to 2 times before defaulting to "no".
    """
    if state.subtask_decision is None:
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
    """Judge the subtasks and track retry attempts."""
    if state.subtask_judgment_retry is None:
        state.subtask_judgment_retry = JudgmentRetry()
    
    result = judge_subtasks(state.task_metadata, state.subtask_metadata)
    state.subtask_judgment = result
    
    # Reset retry counter on pass
    if result.judgment == JudgmentType.PASS:
        state.subtask_judgment_retry.retries = 0
    else:
        state.subtask_judgment_retry.retries += 1
        
        # If we've hit max retries, force a pass
        if state.subtask_judgment_retry.retries >= state.subtask_judgment_retry.max_retries:
            state.subtask_judgment = SubtaskJudgment(
                judgment=JudgmentType.PASS,
                reason="Max retries reached, proceeding with subtasks"
            )
    
    return state

def create_task_node(state: TaskAgentState) -> TaskAgentState:
    create_task(state.task_metadata.task, state.subtask_metadata.subtasks)
    state.confirmed = True
    return state

def ask_about_task_node(state: TaskAgentState) -> TaskAgentState:
    """
    Pause to ask the user for clarification after failed judgment.
    Uses concerns and questions to generate a human-friendly message.
    
    Control flow:
    1. First run:
       - Clears previous user feedback
       - Generates clarification prompt
       - Raises GraphInterrupt for HITL
    2. Resume run:
       - Returns state with user feedback for processing
    """
    if state.user_feedback is None:
        state.user_feedback = None
        prompt = generate_task_clarification_prompt(state.task_metadata, state.task_judgment, "task")
        raise GraphInterrupt(
            Interrupt(
                value=prompt,
                resumable=True
            )
        )
    return state

def retry_task_node(state: TaskAgentState) -> TaskAgentState:
    """
    Process user feedback to refine the task.
    Clears user_feedback after processing.
    """
    result = retry_task_with_feedback(state)
    state.task_metadata = result
    state.user_feedback = None
    return state

def retry_subtasks_node(state: TaskAgentState) -> TaskAgentState:
    """
    Process user feedback to refine the subtasks.
    Clears user_feedback after processing.
    """
    result = retry_subtasks_with_feedback(state)
    state.subtask_metadata = result
    state.user_feedback = None
    return state

def ask_about_subtasks_node(state: TaskAgentState) -> TaskAgentState:
    """
    Pause to ask the user for clarification about subtasks after failed judgment.
    Uses concerns and questions to generate a human-friendly message.
    """
    if state.user_feedback is None:
        state.user_feedback = None
        prompt = generate_task_clarification_prompt(state.subtask_metadata, state.subtask_judgment, "subtasks")
        raise GraphInterrupt(
            Interrupt(
                value=prompt,
                resumable=True
            )
        )
    return state

# Build the graph
builder = StateGraph(TaskAgentState)

builder.add_node("extract_task", RunnableLambda(extract_task_node))
builder.add_node("judge_task", RunnableLambda(judge_task_node))
builder.add_node("ask_to_subtask", RunnableLambda(ask_to_subtask_node))
builder.add_node("ask_about_task", RunnableLambda(ask_about_task_node))
builder.add_node("retry_task", RunnableLambda(retry_task_node))
builder.add_node("generate_subtasks", RunnableLambda(generate_subtasks_node))
builder.add_node("judge_subtasks", RunnableLambda(judge_subtasks_node))
builder.add_node("ask_about_subtasks", RunnableLambda(ask_about_subtasks_node))
builder.add_node("retry_subtasks", RunnableLambda(retry_subtasks_node))
builder.add_node("create_task", RunnableLambda(create_task_node))

builder.set_entry_point("extract_task")

# Graph edges
builder.add_edge("extract_task", "judge_task")
builder.add_conditional_edges("judge_task", lambda s: s.task_judgment.judgment.value, {
    JudgmentType.PASS.value: "ask_to_subtask",
    JudgmentType.FAIL.value: "ask_about_task"
})
builder.add_edge("ask_about_task", "retry_task")
builder.add_edge("retry_task", "judge_task")

builder.add_conditional_edges("ask_to_subtask", lambda s: s.subtask_decision.value, {
    "yes": "generate_subtasks",
})
builder.add_edge("generate_subtasks", "judge_subtasks")
builder.add_conditional_edges("judge_subtasks", lambda s: s.subtask_judgment.judgment.value, {
    JudgmentType.PASS.value: "create_task",
    JudgmentType.FAIL.value: "ask_about_subtasks"
})
builder.add_edge("ask_about_subtasks", "retry_subtasks")
builder.add_edge("retry_subtasks", "judge_subtasks")
builder.add_edge("create_task", END)

# Compile the graph
graph = builder.compile()
