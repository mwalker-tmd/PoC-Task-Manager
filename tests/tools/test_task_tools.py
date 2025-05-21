import pytest
from backend.tools import task_tools

def test_extract_task():
    result = task_tools.extract_task("Do the dishes")
    assert "do the dishes" in result["task"].lower()

def test_analyze_subtasks_with_presentation():
    result = task_tools.analyze_subtasks("Create a presentation")
    assert result["has_subtasks"] is True
    subtasks = result["subtasks"]
    assert len(subtasks) > 0
    assert any(term in subtask.lower() for subtask in subtasks for term in [
        "purpose", "objective", "goal", "aim", "intent", "target", "focus",
        "define", "determine", "identify", "establish", "set"
    ])
    assert any(term in subtask.lower() for subtask in subtasks for term in [
        "structure", "outline", "content", "section", "slide", "visual"
    ])

def test_analyze_subtasks_without_presentation():
    result = task_tools.analyze_subtasks("Go shopping")
    assert result["has_subtasks"] is True
    subtasks = result["subtasks"]
    assert len(subtasks) > 0
    assert any("list" in subtask.lower() or "needed" in subtask.lower() for subtask in subtasks)

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