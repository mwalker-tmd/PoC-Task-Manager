from typing import Union
from backend.tools.task_tools import _make_llm_call, get_client, DEFAULT_MODEL
from backend.types import TaskMetadata, SubtaskMetadata, TaskJudgment
import traceback
from backend.logger import logger
from backend.prompts.task_prompts import TASK_CLARIFICATION_SYSTEM_PROMPT

def generate_task_clarification_prompt(metadata: Union[TaskMetadata, SubtaskMetadata], judgment: TaskJudgment, task_type: str) -> str:
    """
    Use LLM to create a human-facing message asking for clarification or confirmation based on concerns and questions.
    """
    # Handle task-specific content
    task_content = ""
    if isinstance(metadata, TaskMetadata):
        task_content = f"<task>{metadata.task}</task>"
    else:  # SubtaskMetadata
        task_content = f"<subtasks>{chr(10).join(metadata.subtasks)}</subtasks>"

    user_prompt = f"""
    <user_prompt>
        {task_content}
        <judgment>{judgment.judgment}</judgment>
        <reason>{judgment.reason}</reason>

        <concerns>
        {chr(10).join(metadata.concerns) if metadata.concerns else 'None'}
        </concerns>

        <questions>
        {chr(10).join(metadata.questions) if metadata.questions else 'None'}
        </questions>
    </user_prompt>
    """

    try:
        content = _make_llm_call(TASK_CLARIFICATION_SYSTEM_PROMPT.format(task_type=task_type), user_prompt)
        return content
    except Exception as e:
        logger.error(f"Failed to generate {task_type} clarification prompt: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return f"I need some clarification about your {task_type}. Could you please provide more details?"
