from backend.graphs.task_agent import TaskAgentState, extract_task_node, generate_subtasks_node

def test_task_agent_state_optional_input():
    state = TaskAgentState()
    assert state.input is None

def test_extract_task_node():
    state = TaskAgentState(input="Do the dishes")
    result = extract_task_node(state)
    assert "do the dishes" in result["task"].lower()

def test_generate_subtasks_node_with_presentation():
    state = TaskAgentState(task="Create a presentation")
    result = generate_subtasks_node(state)
    subtasks = result["subtasks"]
    print("\nGenerated subtasks:", subtasks)  # Debug output
    assert len(subtasks) > 0
    assert any(term in subtask.lower() for subtask in subtasks for term in [
        "purpose", "objective", "goal", "aim", "intent", "target", "focus",
        "define", "determine", "identify", "establish", "set"
    ])
    assert any(term in subtask.lower() for subtask in subtasks for term in [
        "structure", "outline", "content", "section", "slide", "visual"
    ])

def test_generate_subtasks_node_without_presentation():
    state = TaskAgentState(task="Go shopping")
    result = generate_subtasks_node(state)
    subtasks = result["subtasks"]
    assert len(subtasks) > 0
    assert any("list" in subtask.lower() or "needed" in subtask.lower() for subtask in subtasks) 