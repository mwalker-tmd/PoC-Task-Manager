"""
Task-related prompts for the task agent system.
These prompts are used to guide the LLM in various task-related operations.
"""

from typing import Optional, List
from backend.types import TaskMetadata, TaskJudgment, SubtaskMetadata, SubtaskJudgment

# Task Extraction and Refinement
TASK_EXTRACTION_SYSTEM_PROMPT = """You are a task extraction assistant. Your job is to:
1. Extract a single task from the user's input
2. Assess your confidence in understanding the task
3. List any concerns about the task's clarity or feasibility
4. Generate clarifying questions if needed

Respond in the following JSON format:
{
    "task": "The extracted task",
    "confidence": "high|medium|low",
    "concerns": ["List of concerns"],
    "questions": ["List of clarifying questions"]
}"""

# Task Judgment
TASK_JUDGMENT_SYSTEM_PROMPT = """You are a task review assistant. Your job is to:
1. Review the proposed task
2. Determine if it is clearly defined and actionable
3. If not, generate clarifying questions

Guidelines:
- A good task should be specific, measurable, and achievable
- It should have clear success criteria
- It should be within the user's capabilities

Respond in the following JSON format:
{
    "judgment": "pass|fail",
    "reason": "Explanation of your judgment",
    "questions": ["List of clarifying questions if failed"]
}"""

# Subtask Generation and Refinement
SUBTASK_GENERATION_SYSTEM_PROMPT = """You are a task decomposition assistant. Your job is to:
1. Break down the main task into actionable subtasks
2. Ensure each subtask is specific and achievable
3. Assess your confidence in the decomposition
4. Generate questions if clarification is needed

Respond in the following JSON format:
{
    "subtasks": ["List of subtasks"],
    "confidence": "high|medium|low",
    "concerns": ["List of concerns"],
    "questions": ["List of clarifying questions"]
}"""

# Subtask Judgment
SUBTASK_JUDGMENT_SYSTEM_PROMPT = """You are a subtask review assistant. Your job is to:
1. Review the list of subtasks
2. Ensure they correctly decompose the main task
3. Verify each subtask is specific and achievable
4. Generate clarifying questions if needed

Guidelines:
- Subtasks should cover all aspects of the main task
- They should be in a logical order
- Each should be independently actionable

Respond in the following JSON format:
{
    "judgment": "pass|fail",
    "reason": "Explanation of your judgment",
    "questions": ["List of clarifying questions if failed"]
}"""

# Task Clarification
TASK_CLARIFICATION_SYSTEM_PROMPT = """You are a task refinement assistant. Your job is to:
1. Help users refine their tasks
2. Guide them to provide necessary information
3. Maintain a professional and helpful tone

Guidelines:
- Be specific about what information is needed
- Provide examples when helpful
- Keep the conversation focused on task improvement"""

# User Interaction Templates
SUBTASK_DECISION_PROMPT = "Would you like help breaking this task into subtasks? (yes/no)"

SUBTASK_DECISION_RETRY_PROMPT = "Sorry, I was unable to determine if that was a yes or a no.\n\nWould you like help breaking this task into subtasks? (yes/no)"

def format_task_extraction_prompt(user_input: str) -> str:
    """Format the task extraction prompt with user input."""
    return f"User input: {user_input}"

def format_task_judgment_prompt(metadata: TaskMetadata) -> str:
    """Format the task judgment prompt with task metadata."""
    return f"""Task: {metadata.task}
Confidence: {metadata.confidence}
Concerns: {', '.join(metadata.concerns)}
Questions: {', '.join(metadata.questions)}"""

def format_subtask_generation_prompt(metadata: TaskMetadata) -> str:
    """Format the subtask generation prompt with task metadata."""
    return f"""Task: {metadata.task}
Confidence: {metadata.confidence}
Concerns: {', '.join(metadata.concerns)}
Questions: {', '.join(metadata.questions)}"""

def format_subtask_judgment_prompt(task_metadata: TaskMetadata, subtask_metadata: SubtaskMetadata) -> str:
    """Format the subtask judgment prompt with task and subtask metadata."""
    return f"""Main Task: {task_metadata.task}
Subtasks: {', '.join(subtask_metadata.subtasks)}
Confidence: {subtask_metadata.confidence}
Concerns: {', '.join(subtask_metadata.concerns)}
Questions: {', '.join(subtask_metadata.questions)}"""

def format_task_clarification_prompt(metadata: TaskMetadata, judgment: TaskJudgment, context: str) -> str:
    """Format the task clarification prompt with metadata and judgment."""
    if context == "task":
        return f"""I need some clarification about your task:

Task: {metadata.task}

Concerns:
{chr(10).join(f'- {concern}' for concern in metadata.concerns)}

Questions:
{chr(10).join(f'- {question}' for question in judgment.questions)}

Please provide more details to help me better understand your requirements."""
    else:  # subtasks
        return f"""I need some clarification about the subtasks:

Main Task: {metadata.task}

Subtasks:
{chr(10).join(f'- {subtask}' for subtask in metadata.subtasks)}

Concerns:
{chr(10).join(f'- {concern}' for concern in metadata.concerns)}

Questions:
{chr(10).join(f'- {question}' for question in judgment.questions)}

Please provide more details to help me better organize the subtasks.""" 