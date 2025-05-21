from typing import List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
DEFAULT_MODEL = "gpt-4"

# --- Shared LLM client accessor ---
def get_client():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable. "
            "You can get an API key from https://platform.openai.com/api-keys"
        )
    return OpenAI(api_key=openai_api_key)

# --- LLM-enabled functions ---
def extract_task(state) -> TaskMetadata:
    """
    Use LLM to extract the main task, assess confidence, raise concerns, and generate clarifying questions.
    """
    client = get_client()

    system_msg = (
        "You are an expert task manager assistant."
        " Given a user request, extract the main task, assess how confident you are in your interpretation,"
        " list any concerns or ambiguities, and write any clarification questions you’d ask the user before proceeding."
    )

    user_prompt = f"""
    USER REQUEST:
    {state.input}

    Respond in this JSON format:
    {{
      "task": <string>,
      "confidence": <float 0-1>,
      "concerns": [<string>...],
      "questions": [<string>...]
    }}
    """

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    )

    import json
    try:
        content = response.choices[0].message.content.strip()
        return TaskMetadata(**json.loads(content))
    except Exception as e:
        return TaskMetadata(
            task=state.input.strip(),
            confidence=0.5,
            concerns=["Failed to parse LLM response"],
            questions=[]
        )

def analyze_subtasks(task: str) -> dict:
    """
    Break the main task into subtasks and identify any missing info using an LLM.
    """
    client = get_client()
    prompt = f"""
    Task: "{task}"

    List any subtasks that make up this task, and specify any missing information needed to complete it.
    Provide two lists: one of subtasks, one of missing info.
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You help break down tasks into subtasks and spot missing info."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip()
    lines = content.splitlines()
    subtasks, missing = [], []
    current = None
    for line in lines:
        if "subtask" in line.lower():
            current = subtasks
            continue
        elif "missing" in line.lower():
            current = missing
            continue
        if current is not None and line.strip():
            current.append(line.lstrip("- ").strip())

    return {
        "has_subtasks": bool(subtasks),
        "subtasks": subtasks,
        "missing_info": missing
    }

def clarify(missing_info: List[str]) -> dict:
    """
    Generate clarification questions for missing task info using an LLM.
    """
    client = get_client()
    if not missing_info:
        return {"questions": []}

    prompt = f"""
    A task is missing the following pieces of information: {', '.join(missing_info)}.
    Write a short clarification question for each.
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You generate clarification questions for missing task details."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip()
    questions = [line.lstrip("- ").strip() for line in content.splitlines() if line.strip()]
    return {
        "questions": questions
    }

def review(task: str, subtasks: Optional[List[str]] = None) -> dict:
    """
    Present the task and subtasks back to the user for review.
    """
    return {
        "task": task,
        "subtasks": subtasks or [],
        "message": "Here's what I've extracted. Let me know if you'd like to change anything."
    }

def revise_subtasks(user_feedback: str, subtasks: List[str]) -> dict:
    """
    Accept user feedback and modify subtasks using an LLM.
    """
    client = get_client()
    prompt = f"""
    Current subtasks:
    {chr(10).join(f"- {s}" for s in subtasks)}

    User feedback:
    "{user_feedback}"

    Please return an updated list of subtasks based on the feedback.
    Return them as a plain numbered list.
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that revises task subtasks."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip()
    new_subtasks = [line.split(".", 1)[-1].strip(" ") for line in content.splitlines() if line.strip()]
    return {
        "subtasks": new_subtasks
    }

def judge_task(original_prompt: str, final_task: str) -> dict:
    """
    Check if the task is a reasonable revision of the user's submission.
    """
    client = get_client()
    prompt = f"""
    Original task input:
    "{original_prompt}"

    Extracted main task:
    "{final_task}"

    Is the main task a reasonable interpretation of the user's submission? Respond with "approved" or "needs revision" only.
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert task workflow judge."},
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0].message.content.strip().lower()
    return {
        "status": "approved" if "approved" in result else "needs revision"
    }

def judge_subtasks(original_prompt: str, final_task: str, subtasks: List[str]) -> dict:
    """
    Check if the subtasks are a reasonable decomposition of the original task.
    """
    client = get_client()
    prompt = f"""
    Original task input:
    "{original_prompt}"

    Extracted main task:
    "{final_task}"

    Proposed subtasks:
    {chr(10).join(f"- {s}" for s in subtasks)}

    Are the subtasks a reasonable breakdown of the task? Respond with "approved" or "needs revision" only.
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert task workflow judge."},
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0].message.content.strip().lower()
    return {
        "status": "approved" if "approved" in result else "needs revision"
    }

def save_task_to_db(task: str, subtasks: Optional[List[str]] = None):
    subtasks = subtasks or []
    print(f"[SAVE] Task: {task}\n[SUBTASKS]\n" + chr(10).join(f"- {s}" for s in subtasks))
    return {
        "status": "saved"
    }


# The following functions are stubs for the v2 task agent
def ask_to_subtask(task: str) -> dict:
    """
    Ask the user to select a subtask from a list.
    """
    return {
        "decision": "no"
    }
def create_clarifying_questions(task: str) -> dict:    
    """
    Create clarifying questions for a subtask.
    """
    return {
        "questions": [f"What is the purpose of {subtask}?" for subtask in subtasks]
    }
def receive_clarification_feedback(feedback: str, subtask: str) -> dict:
    """
    Receive feedback on a clarification question.
    """
    return {
        "feedback": feedback
    }
