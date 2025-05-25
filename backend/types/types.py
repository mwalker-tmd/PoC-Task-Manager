from pydantic import BaseModel
from typing import Optional, List, Literal
from enum import Enum

class JudgmentType(str, Enum):
    PASS = "pass"
    FAIL = "fail"

class UserFeedbackRetry(BaseModel):
    retries: int = 0
    max_retries: int = 3

class BaseMetadata(BaseModel):
    confidence: Optional[float] = None
    concerns: Optional[List[str]] = None
    questions: Optional[List[str]] = None

class TaskMetadata(BaseModel):
    task: str
    confidence: float
    concerns: List[str]
    questions: List[str]
    is_subtaskable: bool = False

class TaskJudgment(BaseModel):
    judgment: JudgmentType
    reason: str

class SubtaskMetadata(BaseModel):
    subtasks: List[str]
    confidence: float = 1.0
    concerns: List[str] = []
    questions: List[str] = []

class SubtaskJudgment(BaseModel):
    judgment: JudgmentType
    reason: str

class SubtaskDecision(BaseModel):
    value: Optional[str] = None
    user_feedback_retry: Optional[UserFeedbackRetry] = None

class TaskAgentState(BaseModel):
    input: Optional[str] = None
    task_metadata: Optional[TaskMetadata] = None
    task_judgment: Optional[TaskJudgment] = None
    task_judgment_retry: Optional[UserFeedbackRetry] = None
    subtask_decision: Optional[SubtaskDecision] = None
    subtask_metadata: Optional[SubtaskMetadata] = None
    subtask_judgment: Optional[SubtaskJudgment] = None
    subtask_judgment_retry: Optional[UserFeedbackRetry] = None
    user_feedback: Optional[str] = None
    confirmed: bool = False
