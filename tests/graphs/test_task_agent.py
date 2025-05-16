from backend.graphs.task_agent import TaskAgentState, extract_task_node, analyze_subtasks_node

def test_task_agent_state_optional_input():
    state = TaskAgentState()
    assert state.input is None

def test_extract_task_node():
    state = TaskAgentState(input="Do the dishes")
    result = extract_task_node(state)
    assert result["task"] == "Do the dishes"

def test_analyze_subtasks_node_with_presentation():
    state = TaskAgentState(task="Create a presentation")
    result = analyze_subtasks_node(state)
    assert "Define topic" in result["subtasks"]
    assert "topic" in result["clarification_needed"]

def test_analyze_subtasks_node_without_presentation():
    state = TaskAgentState(task="Go shopping")
    result = analyze_subtasks_node(state)
    assert result["subtasks"] == []
    assert result["clarification_needed"] == [] 