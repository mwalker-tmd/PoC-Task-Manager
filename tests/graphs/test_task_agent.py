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
    judge_task_node,
    process_subtask_decision_node
)
from backend.types import TaskMetadata, SubtaskMetadata, JudgmentType, SubtaskDecision, TaskJudgment, SubtaskJudgment, UserFeedbackRetry
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

def test_ask_to_subtask_node_initial_prompt():
    """Test the initial prompt when no decision exists."""
    state = TaskAgentState()
    with pytest.raises(GraphInterrupt) as exc_info:
        ask_to_subtask_node(state)
    assert "Would you like help breaking this task into subtasks?" in str(exc_info.value)
    assert state.subtask_decision.user_feedback_retry.retries == 0

def test_ask_to_subtask_node_retry_prompt():
    """Test the retry prompt with the 'Sorry' message."""
    state = TaskAgentState(
        subtask_decision=SubtaskDecision(
            value=None,
            user_feedback_retry=UserFeedbackRetry()
        )
    )
    with pytest.raises(GraphInterrupt) as exc_info:
        ask_to_subtask_node(state)
    assert "Sorry, I was unable to determine" in str(exc_info.value)
    assert state.subtask_decision.user_feedback_retry.retries == 1

def test_ask_to_subtask_node_existing_decision():
    """Test that an existing valid decision is preserved."""
    state = TaskAgentState(
        subtask_decision=SubtaskDecision(
            value="yes",
            user_feedback_retry=UserFeedbackRetry()
        ),
        user_feedback="some feedback"
    )
    with pytest.raises(GraphInterrupt) as exc_info:
        ask_to_subtask_node(state)
    assert "Would you like help breaking this task into subtasks?" in str(exc_info.value)
    assert state.subtask_decision.user_feedback_retry.retries == 1

def test_process_subtask_decision_node_valid_yes():
    """Test processing various valid 'yes' responses."""
    valid_yes_responses = ['yes', 'YES', 'Yes', 'y', 'Y', 'true', 'True', 'TRUE', 't', 'T', '1', 'on']
    for response in valid_yes_responses:
        state = TaskAgentState(
            subtask_decision=SubtaskDecision(
                value=None,
                user_feedback_retry=UserFeedbackRetry()
            ),
            user_feedback=response
        )
        result = process_subtask_decision_node(state)
        assert result.subtask_decision.value == "yes"
        assert result.user_feedback is None

def test_process_subtask_decision_node_valid_no():
    """Test processing various valid 'no' responses."""
    valid_no_responses = ['no', 'NO', 'No', 'n', 'N', 'false', 'False', 'FALSE', 'f', 'F', '0', 'off']
    for response in valid_no_responses:
        state = TaskAgentState(
            subtask_decision=SubtaskDecision(
                value=None,
                user_feedback_retry=UserFeedbackRetry()
            ),
            user_feedback=response
        )
        result = process_subtask_decision_node(state)
        assert result.subtask_decision.value == "no"
        assert result.user_feedback is None

def test_process_subtask_decision_node_invalid_response():
    """Test processing an invalid response."""
    state = TaskAgentState(
        subtask_decision=SubtaskDecision(
            value=None,
            user_feedback_retry=UserFeedbackRetry(retries=3)
        ),
        user_feedback="maybe"
    )
    result = process_subtask_decision_node(state)
    assert result.subtask_decision.value == "no"  # Should default to no at max retries
    assert result.user_feedback is None

def test_process_subtask_decision_node_invalid_response_below_max_retries():
    """Test processing an invalid response when below max retries."""
    state = TaskAgentState(
        subtask_decision=SubtaskDecision(
            value=None,
            user_feedback_retry=UserFeedbackRetry(retries=1)
        ),
        user_feedback="maybe"
    )
    result = process_subtask_decision_node(state)
    assert result.subtask_decision.value is None  # Should not set value when below max retries
    assert result.user_feedback is None

def test_process_subtask_decision_node_no_feedback():
    """Test processing when no feedback is present."""
    state = TaskAgentState(
        subtask_decision=SubtaskDecision(
            value=None,
            user_feedback_retry=UserFeedbackRetry()
        )
    )
    result = process_subtask_decision_node(state)
    assert result.subtask_decision.value is None
    assert result.user_feedback is None

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
    with pytest.raises(GraphInterrupt) as exc_info:
        ask_about_task_node(state)
    assert "I need some clarification about your task" in str(exc_info.value)
    assert "Task is vague" in str(exc_info.value)
    assert "Which dishes?" in str(exc_info.value)
    assert state.user_feedback is None

def test_ask_about_task_node_resume():
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
    with pytest.raises(GraphInterrupt) as exc_info:
        ask_about_subtasks_node(state)
    assert "I need some clarification about your subtasks" in str(exc_info.value)
    assert "Unclear where to put dishes" in str(exc_info.value)
    assert "Where should the dishes go?" in str(exc_info.value)
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
                "Dry and put in cabinet"
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
            user_feedback="Put them in the cabinet"
        ),
        subtask_metadata=SubtaskMetadata(
            subtasks=["Fill sink", "Scrub", "Rinse"],
            confidence=0.5,
            concerns=["Unclear where to put dishes"],
            questions=["Where should the dishes go?"]
        )
    )
    result = retry_subtasks_node(state)
    assert isinstance(result.subtask_metadata, SubtaskMetadata)
    assert len(result.subtask_metadata.subtasks) > 0
    assert any("put in cabinet" in subtask.lower() for subtask in result.subtask_metadata.subtasks)
    assert result.subtask_metadata.confidence == 0.9
    assert result.subtask_metadata.concerns == []
    assert result.subtask_metadata.questions == []
    assert result.user_feedback is None

def test_judge_task_node_retry_behavior(mock_openai):
    """Test that judge_task_node properly tracks retries and forces pass after max retries."""
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
    """Test that judge_subtasks_node properly tracks retries and forces pass after max retries."""
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