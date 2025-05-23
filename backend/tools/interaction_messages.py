from typing import Union

def generate_task_clarification_prompt(metadata: Union[TaskMetadata, SubtaskMetadata], judgment: TaskJudgment, task_type: str) -> str:
    """
    Use LLM to create a human-facing message asking for clarification or confirmation based on concerns and questions.
    """
    client = get_client()

    system_msg = """
    <system_prompt>
        You are a task refinement specialist helping users clarify and improve their {task_type}.
        Your job is to write a friendly, conversational message that:
        1. Acknowledges the current state of the task
        2. Clearly presents any concerns or questions that need addressing
        3. Guides the user toward providing the necessary information
        4. Maintains a helpful and professional tone

        Consider the following when crafting your message:
        - If there are concerns, explain why they matter and how addressing them will help
        - If there are questions, present them in a logical order
        - If the judgment is "fail", explain what needs to change to make it pass
        - Keep the message concise but complete

        Your response should be a single, well-structured message that the user can easily understand and respond to.
    </system_prompt>
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

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()
