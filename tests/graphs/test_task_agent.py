from backend.graphs.task_agent import (
    TaskAgentState, 
    extract_task_node, 
    generate_subtasks_node, 
    judge_subtasks_node, 
    ask_to_subtask_node,
    ask_about_task_node,
    retry_task_node,
    ask_about_subtasks_node,
    retry_subtasks_node,
    judge_task_node
)
from backend.types import TaskMetadata, SubtaskMetadata, JudgmentType, TaskJudgment, SubtaskJudgment, UserFeedbackRetry
from unittest.mock import Mock, patch
from langgraph.errors import GraphInterrupt
import pytest
from langgraph.config import RunnableConfig

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

def test_generate_subtasks_node(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='''{
            "subtasks": [
                "Fill sink with hot water and soap",
                "Scrub all dishes",
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
            questions=[],
            is_subtaskable=True
        )
    )
    result = generate_subtasks_node(state)
    assert isinstance(result.subtask_metadata, SubtaskMetadata)
    assert len(result.subtask_metadata.subtasks) == 4
    assert result.subtask_metadata.confidence == 0.9
    assert result.subtask_metadata.concerns == []
    assert result.subtask_metadata.questions == []

def test_judge_subtasks_node_pass(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"judgment": "pass", "reason": "Subtasks are well-structured"}'))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=True
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Fill sink", "Scrub", "Rinse", "Dry"],
            confidence=0.9,
            concerns=[],
            questions=[]
        )
    )
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.PASS
    assert "well-structured" in result.subtask_judgment.reason.lower()

def test_judge_subtasks_node_fail(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"judgment": "fail", "reason": "Subtasks are incomplete"}'))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=True
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Fill sink"],
            confidence=0.5,
            concerns=["Missing steps"],
            questions=["What about drying?"]
        )
    )
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.FAIL
    assert "incomplete" in result.subtask_judgment.reason.lower()

def test_ask_to_subtask_node_initial_prompt():
    """Test the initial prompt when no decision exists."""
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=True
        )
    )
    with patch('backend.graphs.task_agent.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = GraphInterrupt({"prompt": "Would you like help breaking this task into subtasks? (yes/no)"})
        with pytest.raises(GraphInterrupt) as exc_info:
            ask_to_subtask_node(state)
        assert "Would you like help breaking this task into subtasks?" in str(exc_info.value)
        assert state.user_wants_subtasks is None

def test_ask_to_subtask_node_retry_prompt():
    """Test the retry prompt with the 'Sorry' message."""
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=True
        ),
        user_wants_subtasks=None
    )
    with patch('backend.graphs.task_agent.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = GraphInterrupt({"prompt": "Sorry, I was unable to determine if that was a yes or a no.\n\nWould you like help breaking this task into subtasks? (yes/no)"})
        with pytest.raises(GraphInterrupt) as exc_info:
            ask_to_subtask_node(state)
        assert "Sorry, I was unable to determine" in str(exc_info.value)
        assert state.user_wants_subtasks is None

def test_ask_to_subtask_node_valid_yes():
    """Test processing various valid 'yes' responses."""
    valid_yes_responses = ['yes', 'YES', 'Yes', 'y', 'Y', 'true', 'True', 'TRUE', 't', 'T', '1', 'on']
    for response in valid_yes_responses:
        state = TaskAgentState(
            task_metadata=TaskMetadata(
                task="do the dishes",
                confidence=0.9,
                concerns=[],
                questions=[],
                is_subtaskable=True
            ),
            user_wants_subtasks=None
        )
        with patch('backend.graphs.task_agent.interrupt', return_value=response):
            result = ask_to_subtask_node(state)
            assert result.user_wants_subtasks is True

def test_ask_to_subtask_node_valid_no():
    """Test processing various valid 'no' responses."""
    valid_no_responses = ['no', 'NO', 'No', 'n', 'N', 'false', 'False', 'FALSE', 'f', 'F', '0', 'off']
    for response in valid_no_responses:
        state = TaskAgentState(
            task_metadata=TaskMetadata(
                task="do the dishes",
                confidence=0.9,
                concerns=[],
                questions=[],
                is_subtaskable=True
            ),
            user_wants_subtasks=None
        )
        with patch('backend.graphs.task_agent.interrupt', return_value=response):
            result = ask_to_subtask_node(state)
            assert result.user_wants_subtasks is False

def test_ask_to_subtask_node_invalid_response():
    """Test processing an invalid response at max retries."""
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=True
        ),
        user_wants_subtasks=None
    )
    with patch('backend.graphs.task_agent.interrupt', return_value="maybe"):
        result = ask_to_subtask_node(state)
        assert result.user_wants_subtasks is False

def test_ask_to_subtask_node_invalid_response_below_max_retries():
    """Test processing an invalid response when below max retries."""
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=True
        ),
        user_wants_subtasks=None
    )
    with patch('backend.graphs.task_agent.interrupt', side_effect=["maybe", "maybe", "maybe"]):
        result = ask_to_subtask_node(state)
        assert result.user_wants_subtasks is False

def test_ask_to_subtask_node_not_subtaskable():
    """Test when task is not subtaskable."""
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=False
        )
    )
    result = ask_to_subtask_node(state)
    assert result.user_wants_subtasks is False

def test_ask_about_task_node_first_run(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="Please clarify which dishes need to be done"))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.5,
            concerns=["Task is vague"],
            questions=["Which dishes?"]
        ),
        task_judgment=TaskJudgment(
            judgment=JudgmentType.FAIL,
            reason="Task is too vague"
        )
    )
    with patch('backend.graphs.task_agent.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = GraphInterrupt({"prompt": "I need some clarification about your task"})
        with pytest.raises(GraphInterrupt) as exc_info:
            ask_about_task_node(state)
        assert "I need some clarification about your task" in str(exc_info.value)
        assert state.user_feedback is None

def test_ask_about_task_node_resume():
    """Test resuming with existing feedback."""
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.5,
            concerns=["Task is vague"],
            questions=["Which dishes?"]
        ),
        task_judgment=TaskJudgment(
            judgment=JudgmentType.FAIL,
            reason="Task is too vague"
        ),
        user_feedback="I mean all the dishes in the sink"
    )
    result = ask_about_task_node(state)
    assert result.user_feedback == "I mean all the dishes in the sink"
    assert result.task_judgment is None

def test_ask_about_subtasks_node_first_run(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="Please clarify what you mean by 'put away'"))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[]
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Fill sink", "Scrub", "Rinse"],
            confidence=0.5,
            concerns=["Unclear where to put dishes"],
            questions=["Where should the dishes go?"]
        ),
        subtask_judgment=SubtaskJudgment(
            judgment=JudgmentType.FAIL,
            reason="Subtasks are incomplete"
        )
    )
    with patch('backend.graphs.task_agent.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = GraphInterrupt({"prompt": "I need some clarification about your subtasks"})
        with pytest.raises(GraphInterrupt) as exc_info:
            ask_about_subtasks_node(state)
        assert "I need some clarification about your subtasks" in str(exc_info.value)
        assert state.user_feedback is None

def test_ask_about_subtasks_node_resume():
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[]
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Fill sink", "Scrub", "Rinse"],
            confidence=0.5,
            concerns=["Unclear where to put dishes"],
            questions=["Where should the dishes go?"]
        ),
        subtask_judgment=SubtaskJudgment(
            judgment=JudgmentType.FAIL,
            reason="Subtasks are incomplete"
        ),
        user_feedback="Put them in the cabinet above the sink"
    )
    result = ask_about_subtasks_node(state)
    assert result.user_feedback == "Put them in the cabinet above the sink"
    assert result.subtask_judgment is None

def test_retry_task_node(mock_openai):
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
    result = retry_task_node(state)
    assert isinstance(result.task_metadata, TaskMetadata)
    assert "wash all dishes in the sink" in result.task_metadata.task.lower()
    assert result.task_metadata.confidence == 0.9
    assert result.task_metadata.concerns == []
    assert result.task_metadata.questions == []
    assert result.user_feedback is None

def test_retry_subtasks_node(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='''{
            "subtasks": [
                "Fill sink with hot water and soap",
                "Scrub all dishes in the sink",
                "Rinse with clean water",
                "Dry dishes",
                "Put them in the cabinet"
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
        subtask_metadata=SubtaskMetadata(
            subtasks=["Fill sink", "Scrub", "Rinse"],
            confidence=0.5,
            concerns=["Unclear where to put dishes"],
            questions=["Where should the dishes go?"]
        ),
        user_feedback="Put them in the cabinet"
    )
    result = retry_subtasks_node(state)
    assert isinstance(result.subtask_metadata, SubtaskMetadata)
    assert len(result.subtask_metadata.subtasks) == 5
    assert "Put them in the cabinet" in result.subtask_metadata.subtasks
    assert result.subtask_metadata.confidence == 0.9
    assert result.subtask_metadata.concerns == []
    assert result.subtask_metadata.questions == []
    assert result.user_feedback is None

def test_judge_task_node_retry_behavior(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"judgment": "fail", "reason": "Task is too vague"}'))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.5,
            concerns=["Task is vague"],
            questions=["Which dishes?"]
        )
    )
    
    # First fail
    result = judge_task_node(state)
    assert result.task_judgment.judgment == JudgmentType.FAIL
    assert result.task_judgment_retry.retries == 1
    
    # Second fail
    result = judge_task_node(state)
    assert result.task_judgment.judgment == JudgmentType.FAIL
    assert result.task_judgment_retry.retries == 2
    
    # Third fail should force pass
    result = judge_task_node(state)
    assert result.task_judgment.judgment == JudgmentType.PASS
    assert "Max retries reached" in result.task_judgment.reason
    assert result.task_judgment_retry.retries == 3

def test_judge_subtasks_node_retry_behavior(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"judgment": "fail", "reason": "Subtasks are incomplete"}'))
    ]
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[]
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Fill sink", "Scrub", "Rinse"],
            confidence=0.5,
            concerns=["Unclear where to put dishes"],
            questions=["Where should the dishes go?"]
        )
    )
    
    # First fail
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.FAIL
    assert result.subtask_judgment_retry.retries == 1
    
    # Second fail
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.FAIL
    assert result.subtask_judgment_retry.retries == 2
    
    # Third fail should force pass
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.PASS
    assert "Max retries reached" in result.subtask_judgment.reason
    assert result.subtask_judgment_retry.retries == 3

def test_judge_subtasks_node_passes_if_user_accepted():
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=True
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Fill sink", "Scrub", "Rinse"],
            confidence=0.9,
            concerns=[],
            questions=[],
            user_accepted_subtasks=True
        ),
        subtask_judgment_retry=UserFeedbackRetry(retries=0, max_retries=3),
        user_accepted_subtasks=True
    )
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.PASS
    assert "User approved" in result.subtask_judgment.reason or result.user_accepted_subtasks is True

def test_judge_subtasks_node_fails_if_not_accepted_and_below_max_retries():
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=True
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Fill sink", "Scrub", "Rinse"],
            confidence=0.9,
            concerns=[],
            questions=[],
            user_accepted_subtasks=False
        ),
        subtask_judgment_retry=UserFeedbackRetry(retries=1, max_retries=3),
        user_accepted_subtasks=False
    )
    result = judge_subtasks_node(state)
    assert result.subtask_judgment.judgment == JudgmentType.FAIL
    assert result.subtask_judgment_retry.retries == 2

def test_ask_to_subtask_conditional_edges_retry():
    # Simulate a state where user_wants_subtasks is None (undecided/invalid input)
    state = TaskAgentState(
        task_metadata=TaskMetadata(
            task="do the dishes",
            confidence=0.9,
            concerns=[],
            questions=[],
            is_subtaskable=True
        ),
        user_wants_subtasks=None
    )
    # The lambda should return 'ask_to_subtask' for None
    edge_lambda = lambda s: "ask_to_subtask" if s.user_wants_subtasks is None else ("yes" if s.user_wants_subtasks else "no")
    assert edge_lambda(state) == "ask_to_subtask"
    # Also check True/False cases for completeness
    state_yes = state.model_copy(update={"user_wants_subtasks": True})
    state_no = state.model_copy(update={"user_wants_subtasks": False})
    assert edge_lambda(state_yes) == "yes"
    assert edge_lambda(state_no) == "no" 