import pytest
from unittest.mock import Mock, patch
from backend.tools import task_tools
from backend.types import TaskMetadata, TaskAgentState, TaskJudgment, SubtaskMetadata

@pytest.fixture
def mock_openai():
    with patch('backend.tools.task_tools.get_client') as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create = Mock()
        yield mock_client

def test_extract_task_basic(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"task": "do the dishes", "confidence": 0.9, "concerns": [], "questions": []}'))
    ]
    state = TaskAgentState(input="Do the dishes")
    result = task_tools.extract_task(state)
    assert isinstance(result, TaskMetadata)
    assert "do the dishes" in result.task.lower()
    assert isinstance(result.confidence, float)
    assert 0 <= result.confidence <= 1
    assert isinstance(result.concerns, list)
    assert isinstance(result.questions, list)

def test_extract_task_empty_input(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"task": "", "confidence": 0.5, "concerns": [], "questions": []}'))
    ]
    state = TaskAgentState(input="")
    result = task_tools.extract_task(state)
    assert isinstance(result, TaskMetadata)
    assert result.task == ""
    assert 0 <= result.confidence <= 1
    assert isinstance(result.concerns, list)
    assert isinstance(result.questions, list)

def test_extract_task_error_handling(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="invalid json"))
    ]
    state = TaskAgentState(input="Do the dishes")
    result = task_tools.extract_task(state)
    assert isinstance(result, TaskMetadata)
    assert result.task == "Do the dishes"
    assert result.confidence == 0.5
    assert "Failed to parse LLM response" in result.concerns
    assert result.questions == []

def test_judge_task_pass(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"judgment": "pass", "reason": "Task is clear"}'))
    ]
    metadata = TaskMetadata(
        task="Do the dishes",
        confidence=0.9,
        concerns=[],
        questions=[]
    )
    result = task_tools.judge_task(metadata)
    assert isinstance(result, TaskJudgment)
    assert result.judgment == "pass"
    assert isinstance(result.reason, str)

def test_judge_task_low_confidence(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"judgment": "fail", "reason": "Low confidence and vague task"}'))
    ]
    metadata = TaskMetadata(
        task="Do something",
        confidence=0.3,
        concerns=["Task is vague"],
        questions=["What exactly needs to be done?"]
    )
    result = task_tools.judge_task(metadata)
    assert isinstance(result, TaskJudgment)
    assert result.judgment == "fail"
    assert "confidence" in result.reason.lower() or "vague" in result.reason.lower()

def test_judge_task_error_handling(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="invalid json"))
    ]
    metadata = TaskMetadata(
        task="Do the dishes",
        confidence=0.9,
        concerns=[],
        questions=[]
    )
    result = task_tools.judge_task(metadata)
    assert isinstance(result, TaskJudgment)
    assert result.judgment == "fail"
    assert "unable to parse" in result.reason.lower()

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

def test_generate_subtasks_basic(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='''{
            "subtasks": [
                "Fill sink with hot water",
                "Add dish soap",
                "Scrub dishes",
                "Rinse and dry"
            ],
            "missing_info": [
                "Water temperature preference",
                "Drying method preference"
            ]
        }'''))
    ]
    metadata = TaskMetadata(
        task="Do the dishes",
        confidence=0.9,
        concerns=[],
        questions=[]
    )
    result = task_tools.generate_subtasks(metadata)
    assert isinstance(result, SubtaskMetadata)
    assert len(result.subtasks) > 0
    assert any("fill" in subtask.lower() for subtask in result.subtasks)
    assert any("scrub" in subtask.lower() for subtask in result.subtasks)
    assert len(result.missing_info) > 0
    assert any("temperature" in info.lower() for info in result.missing_info)

def test_generate_subtasks_error_handling(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="invalid json"))
    ]
    metadata = TaskMetadata(
        task="Do the dishes",
        confidence=0.9,
        concerns=[],
        questions=[]
    )
    result = task_tools.generate_subtasks(metadata)
    assert isinstance(result, SubtaskMetadata)
    assert result.subtasks == []
    assert "Unable to parse" in result.missing_info[0] 