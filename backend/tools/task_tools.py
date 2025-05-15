from typing import List, Optional

def extract_task(prompt: str) -> dict:
    """
    Extract the main task from a user's prompt.
    """
    # Simplified stub; will later use LLM to parse
    return {
        "task": prompt.strip(),  # naive placeholder
        "context": "User-provided input"
    }

def analyze_subtasks(task: str) -> dict:
    """
    Determine if the task has subtasks, and list them if possible.
    """
    # This would later use LLM logic or few-shot prompt to break down
    if "presentation" in task.lower():
        subtasks = [
            "Define topic",
            "Create slides",
            "Practice delivery"
        ]
    else:
        subtasks = []

    return {
        "has_subtasks": bool(subtasks),
        "subtasks": subtasks,
        "missing_info": ["topic"] if "presentation" in task.lower() else []
    }

def clarify(missing_info: List[str]) -> dict:
    """
    Prompt the user for missing details.
    """
    questions = []
    for item in missing_info:
        questions.append(f"Can you clarify the {item}?")

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
        "message": "Here’s what I’ve extracted. Let me know if you’d like to change anything."
    }
