from backend.graphs.task_agent import TaskAgentState, extract_task_node, generate_subtasks_node, judge_subtasks_node, ask_to_subtask_node
from backend.types import TaskMetadata, SubtaskMetadata, JudgmentType, SubtaskDecision
from unittest.mock import Mock
from langgraph.errors import GraphInterrupt
import pytest

def test_task_agent_state_optional_input():
    state = TaskAgentState()
    assert state.input is None

def test_extract_task_node(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"task": "do the dishes", "confidence": 0.9, "concerns": [], "questions": []}'))
    ]
    state = TaskAgentState(input="Do the dishes")
    result = extract_task_node(state)
    assert isinstance(result.task_metadata, TaskMetadata)
    assert "do the dishes" in result.task_metadata.task.lower()
    assert isinstance(result.task_metadata.confidence, float)
    assert isinstance(result.task_metadata.concerns, list)
    assert isinstance(result.task_metadata.questions, list)

def test_generate_subtasks_node_with_presentation(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='''{
            "subtasks": [
                "Define presentation objectives and target audience",
                "Create presentation outline and structure",
                "Design visual elements and slides",
                "Prepare speaking notes and transitions"
            ],
            "confidence": 0.9,
            "concerns": [],
            "questions": []
        }'''))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="Create a presentation",
            confidence=0.9,
            concerns=[],
            questions=[]
        )
    )
    result = generate_subtasks_node(state)
    assert isinstance(result.subtask_metadata, SubtaskMetadata)
    subtasks = result.subtask_metadata.subtasks
    assert len(subtasks) > 0
    assert any(term in subtask.lower() for subtask in subtasks for term in [
        "purpose", "objective", "goal", "aim", "intent", "target", "focus",
        "define", "determine", "identify", "establish", "set"
    ])
    assert any(term in subtask.lower() for subtask in subtasks for term in [
        "structure", "outline", "content", "section", "slide", "visual"
    ])
    assert result.subtask_metadata.confidence == 0.9
    assert result.subtask_metadata.concerns == []
    assert result.subtask_metadata.questions == []

def test_generate_subtasks_node_without_presentation(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='''{
            "subtasks": [
                "Create shopping list",
                "Check store hours and location",
                "Gather reusable bags",
                "Plan shopping route"
            ],
            "confidence": 0.9,
            "concerns": [],
            "questions": []
        }'''))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="Go shopping",
            confidence=0.9,
            concerns=[],
            questions=[]
        )
    )
    result = generate_subtasks_node(state)
    assert isinstance(result.subtask_metadata, SubtaskMetadata)
    subtasks = result.subtask_metadata.subtasks
    assert len(subtasks) > 0
    assert any("list" in subtask.lower() or "needed" in subtask.lower() for subtask in subtasks)
    assert result.subtask_metadata.confidence == 0.9
    assert result.subtask_metadata.concerns == []
    assert result.subtask_metadata.questions == []

def test_generate_subtasks_error_handling(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="invalid json"))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="Do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[]
        )
    )
    result = generate_subtasks_node(state)
    assert isinstance(result.subtask_metadata, SubtaskMetadata)
    assert result.subtask_metadata.subtasks == []
    assert "Unable to parse" in result.subtask_metadata.concerns[0]
    assert result.subtask_metadata.confidence == 0.0
    assert result.subtask_metadata.questions == []

def test_judge_subtasks_node_pass(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"judgment": "pass", "reason": "Subtasks are well-structured"}'))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="Create a presentation",
            confidence=0.9,
            concerns=[],
            questions=[]
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Define objectives", "Create outline", "Design slides"],
            missing_info=[]
        )
    )
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.PASS

def test_judge_subtasks_node_fail(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"judgment": "fail", "reason": "Subtasks are incomplete"}'))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="Create a presentation",
            confidence=0.9,
            concerns=[],
            questions=[]
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Define objectives"],
            missing_info=[]
        )
    )
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.FAIL

def test_judge_subtasks_node_low_confidence(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"judgment": "fail", "reason": "Low confidence in subtask generation and unclear requirements"}'))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="Create a presentation",
            confidence=0.9,
            concerns=[],
            questions=[]
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Define objectives", "Create outline", "Design slides"],
            confidence=0.4,
            concerns=["Unclear presentation requirements"],
            questions=["What is the target audience?", "What is the presentation duration?"]
        )
    )
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.FAIL
    assert "confidence" in result.subtask_judgment.reason.lower()
    assert "unclear" in result.subtask_judgment.reason.lower()

def test_ask_to_subtask_node_first_prompt():
    state = TaskAgentState()
    with pytest.raises(GraphInterrupt) as exc_info:
        ask_to_subtask_node(state)
    assert "Would you like help breaking this task into subtasks?" in str(exc_info.value)
    assert state.subtask_decision.retries == 1

def test_ask_to_subtask_node_retry():
    state = TaskAgentState(subtask_decision=SubtaskDecision(value=None, retries=1))
    with pytest.raises(GraphInterrupt) as exc_info:
        ask_to_subtask_node(state)
    assert "Sorry, I was unable to determine" in str(exc_info.value)
    assert state.subtask_decision.retries == 2

def test_ask_to_subtask_node_max_retries():
    state = TaskAgentState(subtask_decision=SubtaskDecision(value=None, retries=2))
    result = ask_to_subtask_node(state)
    assert result.subtask_decision.value == "no"
    assert result.subtask_decision.retries == 3

def test_ask_to_subtask_node_with_yes():
    state = TaskAgentState(subtask_decision=SubtaskDecision(value="yes", retries=0))
    result = ask_to_subtask_node(state)
    assert result.subtask_decision.value == "yes"
    assert result.subtask_decision.retries == 0

def test_ask_to_subtask_node_with_no():
    state = TaskAgentState(subtask_decision=SubtaskDecision(value="no", retries=0))
    result = ask_to_subtask_node(state)
    assert result.subtask_decision.value == "no"
    assert result.subtask_decision.retries == 0 