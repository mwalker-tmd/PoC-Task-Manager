from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class JudgmentType(str, Enum):
    PASS = "pass"
    FAIL = "fail"

class BaseMetadata(BaseModel):
    confidence: Optional[float] = None
    concerns: Optional[List[str]] = None
    questions: Optional[List[str]] = None

class TaskMetadata(BaseMetadata):
    task: Optional[str] = None

class TaskJudgment(BaseModel):
    judgment: JudgmentType
    reason: Optional[str] = None

class SubtaskMetadata(BaseMetadata):
    subtasks: List[str]

class SubtaskJudgment(BaseModel):
    judgment: JudgmentType
    reason: str

class SubtaskDecision(BaseModel):
    value: Optional[str] = None  # "yes" or "no"
    retries: int = 0

class JudgmentRetry(BaseModel):
    """Tracks retry attempts for task or subtask judgments."""
    retries: int = 0
    max_retries: int = 3

class TaskAgentState(BaseModel):
    """State for the task agent graph."""
    input: Optional[str] = None
    task_metadata: Optional[TaskMetadata] = None
    task_judgment: Optional[TaskJudgment] = None
    subtask_decision: Optional[SubtaskDecision] = None
    subtask_metadata: Optional[SubtaskMetadata] = None
    subtask_judgment: Optional[SubtaskJudgment] = None
    confirmed: bool = False
    user_feedback: Optional[str] = None
    task_judgment_retry: Optional[JudgmentRetry] = None
    subtask_judgment_retry: Optional[JudgmentRetry] = None
