import pytest
from backend.tools import task_tools
from backend.types import TaskMetadata, TaskAgentState

def test_extract_task():
    state = TaskAgentState(input="Do the dishes")
    result = task_tools.extract_task(state)
    assert isinstance(result, TaskMetadata)
    assert "do the dishes" in result.task.lower()
    assert isinstance(result.confidence, float)
    assert isinstance(result.concerns, list)
    assert isinstance(result.questions, list)

def test_clarify():
    result = task_tools.clarify(["topic", "deadline"])
    questions = result["questions"]
    assert len(questions) == 2
    assert any("topic" in question.lower() for question in questions)
    assert any(term in question.lower() for question in questions for term in ["deadline", "time", "when", "due date", "completion"])

def test_review():
    result = task_tools.review("Do the dishes", ["Fill sink", "Scrub"])
    assert result["task"] == "Do the dishes"
    assert "Fill sink" in result["subtasks"]
    assert "Let me know if you'd like to change anything" in result["message"] 