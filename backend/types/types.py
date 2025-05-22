from pydantic import BaseModel
from typing import Optional, List

class TaskMetadata(BaseModel):
    task: Optional[str] = None
    confidence: Optional[float] = None
    concerns: Optional[List[str]] = None
    questions: Optional[List[str]] = None

class TaskJudgment(BaseModel):
    judgment: str  # 'pass' or 'fail'
    reason: Optional[str] = None

class SubtaskMetadata(BaseModel):
    subtasks: List[str]
    missing_info: List[str]

class TaskAgentState(BaseModel):
    input: Optional[str] = None
    task_metadata: Optional[TaskMetadata] = None
    subtask_metadata: Optional[SubtaskMetadata] = None
    user_feedback: Optional[str] = None
    subtask_decision: Optional[str] = None
    clarification_questions: Optional[List[str]] = None
    confirmed: Optional[bool] = False
    task_judgment: Optional[TaskJudgment] = None
    subtask_judgment: Optional[str] = None
