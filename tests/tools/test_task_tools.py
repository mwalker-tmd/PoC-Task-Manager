import pytest
from unittest.mock import Mock, patch
from backend.tools import task_tools
from backend.types import TaskMetadata, TaskAgentState, TaskJudgment, SubtaskMetadata, JudgmentType
from fastapi import HTTPException

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
    assert result.confidence == 0.0
    assert "Unable to parse task extraction response" in result.concerns
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
    assert result.judgment == JudgmentType.PASS
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
    assert result.judgment == JudgmentType.FAIL
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
    assert result.judgment == JudgmentType.FAIL
    assert "unable to parse" in result.reason.lower()

def test_generate_subtasks_basic(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='''{
            "subtasks": [
                "Fill sink with hot water",
                "Add dish soap",
                "Scrub dishes",
                "Rinse and dry"
            ],
            "confidence": 0.9,
            "concerns": [],
            "questions": []
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
    assert result.confidence == 0.9
    assert result.concerns == []
    assert result.questions == []

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
    assert result.confidence == 0.0
    assert "Unable to parse subtask generation response" in result.concerns
    assert result.questions == []

def test_retry_task_with_feedback_basic(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"task": "wash all dishes in the sink", "confidence": 0.9, "concerns": [], "questions": []}'))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.5,
            concerns=["Task is vague"],
            questions=["Which dishes?"]
        ),
        user_feedback="I mean all the dishes in the sink"
    )
    result = task_tools.retry_task_with_feedback(state)
    assert isinstance(result, TaskMetadata)
    assert "wash all dishes in the sink" in result.task.lower()
    assert result.confidence == 0.9
    assert result.concerns == []
    assert result.questions == []

def test_retry_task_with_feedback_error_handling(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="invalid json"))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.5,
            concerns=["Task is vague"],
            questions=["Which dishes?"]
        ),
        user_feedback="I mean all the dishes in the sink"
    )
    result = task_tools.retry_task_with_feedback(state)
    assert isinstance(result, TaskMetadata)
    assert result.task == "do the dishes"  # Should keep original task on error
    assert result.confidence == 0.0
    assert "Unable to parse task refinement response" in result.concerns
    assert result.questions == []

def test_retry_subtasks_with_feedback_basic(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='''{
            "subtasks": [
                "Fill sink with hot water and soap",
                "Scrub all dishes in the sink",
                "Rinse with clean water",
                "Dry and put away"
            ],
            "confidence": 0.9,
            "concerns": [],
            "questions": []
        }'''))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[]
        ),
        user_feedback="Include drying and putting away"
    )
    result = task_tools.retry_subtasks_with_feedback(state)
    assert isinstance(result, SubtaskMetadata)
    assert len(result.subtasks) > 0
    assert any("dry" in subtask.lower() for subtask in result.subtasks)
    assert any("put away" in subtask.lower() for subtask in result.subtasks)
    assert result.confidence == 0.9
    assert result.concerns == []
    assert result.questions == []

def test_retry_subtasks_with_feedback_error_handling(mock_openai):
    mock_openai.chat.completions.create.side_effect = Exception("Invalid JSON response")
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[]
        ),
        user_feedback="Include drying and putting away"
    )
    with pytest.raises(HTTPException) as exc_info:
        task_tools.retry_subtasks_with_feedback(state)
    assert "retry_subtasks_with_feedback failed" in str(exc_info.value)

def test_create_task_basic():
    result = task_tools.create_task("Do the dishes", ["Fill sink", "Scrub", "Rinse"])
    assert result["status"] == "saved"
    assert result["task"] == "Do the dishes"
    assert len(result["subtasks"]) == 3
    assert "Fill sink" in result["subtasks"]
    assert "Scrub" in result["subtasks"]
    assert "Rinse" in result["subtasks"]

def test_create_task_no_subtasks():
    result = task_tools.create_task("Do the dishes")
    assert result["status"] == "saved"
    assert result["task"] == "Do the dishes"
    assert result["subtasks"] == [] 