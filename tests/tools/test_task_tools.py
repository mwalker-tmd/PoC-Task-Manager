import pytest
from backend.tools import task_tools

def test_extract_task():
    result = task_tools.extract_task("Do the dishes")
    assert result["task"] == "Do the dishes"
    assert result["context"] == "User-provided input"

def test_analyze_subtasks_with_presentation():
    result = task_tools.analyze_subtasks("Create a presentation")
    assert result["has_subtasks"] is True
    assert "Define topic" in result["subtasks"]
    assert "topic" in result["missing_info"]

def test_analyze_subtasks_without_presentation():
    result = task_tools.analyze_subtasks("Go shopping")
    assert result["has_subtasks"] is False
    assert result["subtasks"] == []
    assert result["missing_info"] == []

def test_clarify():
    result = task_tools.clarify(["topic", "deadline"])
    assert "Can you clarify the topic?" in result["questions"]
    assert "Can you clarify the deadline?" in result["questions"]

def test_review():
    result = task_tools.review("Do the dishes", ["Fill sink", "Scrub"])
    assert result["task"] == "Do the dishes"
    assert "Fill sink" in result["subtasks"]
    assert "Here’s what I’ve extracted" in result["message"] 