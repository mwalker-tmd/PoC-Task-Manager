from typing import List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from backend.types import TaskMetadata, TaskJudgment, SubtaskMetadata, SubtaskJudgment
from fastapi import HTTPException
from backend.logger import initialize_logger
from backend.prompts.task_prompts import (
    TASK_EXTRACTION_SYSTEM_PROMPT,
    TASK_JUDGMENT_SYSTEM_PROMPT,
    SUBTASK_GENERATION_SYSTEM_PROMPT,
    SUBTASK_JUDGMENT_SYSTEM_PROMPT,
    TASK_CLARIFICATION_SYSTEM_PROMPT,
    SUBTASK_DECISION_PROMPT,
    SUBTASK_DECISION_RETRY_PROMPT
)

# Load environment variables from .env file
load_dotenv()

# Initialize logger after environment variables are loaded
initialize_logger()

# --- Constants ---
DEFAULT_MODEL = "gpt-4.1"

# --- Shared LLM client accessor ---
def get_client():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable. "
            "You can get an API key from https://platform.openai.com/api-keys"
        )
    return OpenAI(api_key=openai_api_key)

def _make_llm_call(system_msg: str, user_prompt: str) -> dict:
    """
    Helper function to make OpenAI API calls.
    
    Args:
        system_msg: The system message for the API call
        user_prompt: The user prompt for the API call
    
    Returns:
        The parsed JSON response from the API
    """
    client = get_client()
    
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)

def extract_task(state) -> TaskMetadata:
    """
    Use LLM to extract the main task, assess confidence, raise concerns, and generate clarifying questions.
    """
    user_prompt = f"""
    <user_prompt>
        {state.input}
    </user_prompt>
    """

    try:
        content = _make_llm_call(TASK_EXTRACTION_SYSTEM_PROMPT, user_prompt)
        return TaskMetadata(**content)
    except Exception as e:
        return TaskMetadata(
            task=state.input.strip(),
            confidence=0.0,
            concerns=["Unable to parse task extraction response"],
            questions=[],
            is_subtaskable=False
        )

def judge_task(metadata: TaskMetadata) -> TaskJudgment:
    """
    Determine if the extracted task is clearly defined and actionable.
    Uses task confidence, concerns, and clarification questions as context.
    """
    user_prompt = f"""
    <user_prompt>
        <task>{metadata.task}</task>
        <confidence>{metadata.confidence}</confidence>

        <concerns>
        {chr(10).join(metadata.concerns) if metadata.concerns else 'None'}
        </concerns>

        <questions>
        {chr(10).join(metadata.questions) if metadata.questions else 'None'}
        </questions>
    </user_prompt>
    """

    try:
        content = _make_llm_call(TASK_JUDGMENT_SYSTEM_PROMPT, user_prompt)
        return TaskJudgment(**content)
    except Exception:
        return TaskJudgment(
            judgment="fail",
            reason="Task judgment failed: unable to parse task judgment response."
        )

def judge_subtasks(metadata: TaskMetadata, subtasks: SubtaskMetadata) -> SubtaskJudgment:
    """
    Evaluate whether the generated subtasks represent a complete and logical decomposition of the main task.
    """
    user_prompt = f"""
    <user_prompt>
        <task>{metadata.task}</task>

        <subtasks>
        {chr(10).join(subtasks.subtasks) if subtasks.subtasks else 'None'}
        </subtasks>

        <confidence>{subtasks.confidence}</confidence>

        <concerns>
        {chr(10).join(subtasks.concerns) if subtasks.concerns else 'None'}
        </concerns>

        <questions>
        {chr(10).join(subtasks.questions) if subtasks.questions else 'None'}
        </questions>
    </user_prompt>
    """

    try:
        content = _make_llm_call(SUBTASK_JUDGMENT_SYSTEM_PROMPT, user_prompt)
        return SubtaskJudgment(**content)
    except Exception:
        return SubtaskJudgment(
            judgment="fail",
            reason="Subtask judgment failed: unable to parse subtask judgment response."
        )

def save_task_to_db(task: str, subtasks: Optional[List[str]] = None):
    subtasks = subtasks or []
    print(f"[SAVE] Task: {task}\n[SUBTASKS]\n" + chr(10).join(f"- {s}" for s in subtasks))
    return {
        "status": "saved"
    }

def generate_subtasks(metadata: TaskMetadata) -> SubtaskMetadata:
    """
    Use LLM to propose subtasks for a given task and identify missing information.
    """
    user_prompt = f"""
    <user_prompt>
        {metadata.task}
    </user_prompt>
    """

    try:
        content = _make_llm_call(SUBTASK_GENERATION_SYSTEM_PROMPT, user_prompt)
        return SubtaskMetadata(**content)
    except Exception:
        return SubtaskMetadata(
            subtasks=[],
            confidence=0.0,
            concerns=["Unable to parse subtask generation response"],
            questions=[]
        )

def create_task(task: str, subtasks: Optional[List[str]] = None) -> dict:
    """
    Create a new task with optional subtasks.
    Currently just logs the task and subtasks.
    TODO: Implement actual task storage
    """
    subtasks = subtasks or []
    print(f"[SAVE] Task: {task}\n[SUBTASKS]\n" + chr(10).join(f"- {s}" for s in subtasks))
    return {
        "status": "saved",
        "task": task,
        "subtasks": subtasks
    }

def generate_task_clarification_prompt(metadata, judgment, context_type: str) -> str:
    """
    Generate a human-friendly prompt for task/subtask clarification.
    Uses concerns and questions from metadata to create a clear message.
    """
    concerns = metadata.concerns or []
    questions = metadata.questions or []

    lines = []

    # Determine the goal of the interaction
    if not concerns and not questions:
        if context_type == "subtasks":
            lines.append("I've broken down your task into the following subtasks:")
            lines.append("\n" + "\n".join(f"- {subtask}" for subtask in metadata.subtasks))
            lines.append("\nAre these subtasks acceptable? If not, please let me know what changes you'd like to make.")
        else:
            lines.append(f"Here's what I came up with for your {context_type}. Does this look right to you?")
    else:
        lines.append(f"I need your help clarifying your {context_type}.")

        if concerns:
            lines.append("\nHere are a few concerns I have:")
            lines.extend(f"- {c}" for c in concerns)

        if questions:
            lines.append("\nCould you please clarify:")
            lines.extend(f"- {q}" for q in questions)

    return "\n".join(lines).strip()

def retry_task_with_feedback(state) -> TaskMetadata:
    """
    Use LLM to refine the task based on user feedback.
    """
    user_prompt = f"""
    <user_prompt>
        <original_task>{state.task_metadata.task}</original_task>
        <user_feedback>{state.user_feedback}</user_feedback>
    </user_prompt>
    """

    try:
        content = _make_llm_call(TASK_CLARIFICATION_SYSTEM_PROMPT, user_prompt)
        return TaskMetadata(**content)
    except Exception as e:
        return TaskMetadata(
            task=state.task_metadata.task,
            confidence=0.0,
            concerns=["Unable to parse task refinement response"],
            questions=[]
        )

def retry_subtasks_with_feedback(state) -> SubtaskMetadata:
    """
    Use LLM to refine subtasks based on user feedback.
    """
    original_prompt = f"<original_prompt>{state.last_user_message}</original_prompt>" if state.last_user_message else ""

    # Handle case when subtask_metadata is None
    original_subtasks = state.subtask_metadata.subtasks if state.subtask_metadata else []

    user_msg = f"""
    <user_prompt>
    <task>{state.task_metadata.task}</task>

    <original_subtasks>
    {chr(10).join(original_subtasks)}
    </original_subtasks>

    <user_feedback>{state.user_feedback}</user_feedback>

    {original_prompt}
    </user_prompt>
    """

    try:
        content = _make_llm_call(SUBTASK_DECISION_PROMPT, user_msg)
        return SubtaskMetadata(**content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retry_subtasks_with_feedback failed: {str(e)}")
