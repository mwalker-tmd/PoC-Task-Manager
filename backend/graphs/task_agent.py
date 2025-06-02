from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda
from typing import Optional, List
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

from backend.types import TaskMetadata, TaskJudgment, JudgmentType, TaskAgentState, UserFeedbackRetry, SubtaskJudgment
from backend.tools import (
    extract_task,
    judge_task,
    generate_subtasks,
    judge_subtasks,
    create_task,
    retry_task_with_feedback,
    retry_subtasks_with_feedback
)
from backend.tools.interaction_messages import generate_task_clarification_prompt
#from backend.tools.task_tools import generate_task_clarification_prompt
from backend.logger import logger

def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).
    
    True values are 'y', 'yes', 't', 'true', 'on', and '1'.
    False values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")

# Node definitions
def extract_task_node(state: TaskAgentState) -> TaskAgentState:
    """
    Extract the main task from user input and reset state for a new task.
    """
    # Reset state for a new task
    state.user_wants_subtasks = None
    state.user_accepted_subtasks = None
    state.user_feedback = None
    state.last_user_message = None
    
    result = extract_task(state)
    state.task_metadata = result
    return state

def judge_task_node(state: TaskAgentState) -> TaskAgentState:
    """Judge the task and track retry attempts."""
    if state.task_judgment_retry is None:
        state.task_judgment_retry = UserFeedbackRetry()
    
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
    if state.task_metadata.is_subtaskable is False:
        state.user_wants_subtasks = False
        return state

    main_prompt = "Would you like help breaking this task into subtasks? (yes/no)"
    retries = 0
    max_retries = 2

    while state.user_wants_subtasks is None and retries < max_retries:
        if retries == 0:
            logger.debug("ask_to_subtask_node: First Pass")
            prompt_message = main_prompt
        else:
            logger.debug("ask_to_subtask_node: Iterative Pass")
            prompt_message = "Sorry, I was unable to determine if that was a yes or a no.\n\n" + \
                main_prompt
        user_input = interrupt({"prompt": prompt_message})
        logger.debug("ask_to_subtask_node: user_input = %s", user_input)

        try:
            state.user_wants_subtasks = strtobool(user_input)
        except ValueError:
            logger.debug("ask_to_subtask_node: Invalid input: %s", user_input)
            retries += 1
            if retries >= max_retries:
                state.user_wants_subtasks = False
            continue
        break
    return state


def generate_subtasks_node(state: TaskAgentState) -> TaskAgentState:
    result = generate_subtasks(state.task_metadata)
    state.subtask_metadata = result
    return state

def judge_subtasks_node(state: TaskAgentState) -> TaskAgentState:
    """Judge the subtasks and track retry attempts."""
    if state.subtask_judgment_retry is None:
        state.subtask_judgment_retry = UserFeedbackRetry()
    
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
    subtasks = state.subtask_metadata.subtasks if state.subtask_metadata else []
    create_task(state.task_metadata.task, subtasks)
    state.task_creation_confirmed = True
    return state

def ask_about_task_node(state: TaskAgentState) -> TaskAgentState:
    """
    Pause to ask the user for clarification after failed judgment.
    Uses concerns and questions to generate a human-friendly message.
    """
    logger.debug("Entering ask_about_task_node ...")
    if state.user_feedback is None:
        user_message = generate_task_clarification_prompt(state.task_metadata, state.task_judgment, "task")
        state.last_user_message = user_message
        user_input = interrupt({"prompt": user_message})
        logger.debug("ask_about_task_node: user_input = %s", user_input)
        state.user_feedback = user_input
    
    state.task_judgment = None
    logger.debug("ask_about_task_node: user_feedback = %s", state.user_feedback)
    logger.debug("Exiting ask_about_task_node ...")
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
    # Preserve the user_accepted_subtasks field from the result
    state.user_accepted_subtasks = result.user_accepted_subtasks
    state.subtask_metadata = result
    state.user_feedback = None
    return state

def ask_about_subtasks_node(state: TaskAgentState) -> TaskAgentState:
    """
    Pause to ask the user for clarification about subtasks after failed judgment.
    Uses concerns and questions to generate a human-friendly message.
    """
    logger.debug("Entering ask_about_subtask_node ...")
    if state.user_feedback is None:
        user_message = generate_task_clarification_prompt(state.subtask_metadata, state.subtask_judgment, "subtasks")
        state.last_user_message = user_message
        user_input = interrupt({"prompt": user_message})
        logger.debug("ask_about_subtask_node: user_input = %s", user_input)
        state.user_feedback = user_input

    state.subtask_judgment = None
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

builder.add_conditional_edges("ask_to_subtask", lambda s: "yes" if s.user_wants_subtasks else "no", {
    "yes": "generate_subtasks",
    "no": "create_task",
    None: "ask_to_subtask"
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
#checkpointer = InMemorySaver()
#graph = builder.compile(checkpointer=checkpointer)
graph = builder.compile()