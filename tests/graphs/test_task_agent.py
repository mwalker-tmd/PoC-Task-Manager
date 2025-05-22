from backend.graphs.task_agent import TaskAgentState, extract_task_node, generate_subtasks_node
from backend.types import TaskMetadata, SubtaskMetadata
from unittest.mock import Mock

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
            "missing_info": [
                "Presentation duration",
                "Technical requirements"
            ]
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
    assert len(result.subtask_metadata.missing_info) > 0

def test_generate_subtasks_node_without_presentation(mock_openai):
    mock_openai.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='''{
            "subtasks": [
                "Create shopping list",
                "Check store hours and location",
                "Gather reusable bags",
                "Plan shopping route"
            ],
            "missing_info": [
                "Budget constraints",
                "Store preferences"
            ]
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
    assert len(result.subtask_metadata.missing_info) > 0

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
    assert "Unable to parse" in result.subtask_metadata.missing_info[0] 