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
    """
    Metadata about a set of subtasks, including the LLM's assessment of user acceptance.
    
    Attributes:
        subtasks: List of subtask strings
        confidence: LLM's confidence in the subtask breakdown (0.0 to 1.0)
        concerns: List of concerns about the subtasks
        questions: List of clarifying questions about the subtasks
        user_accepted_subtasks: LLM's interpretation of whether the user accepted these subtasks
                               in the current iteration. This is updated each time the LLM processes
                               user feedback.
    """
    subtasks: List[str]
    confidence: float = 1.0
    concerns: List[str] = []
    questions: List[str] = []
    user_accepted_subtasks: bool = False

class SubtaskJudgment(BaseModel):
    judgment: JudgmentType
    reason: str

class TaskAgentState(BaseModel):
    """
    The state of the task agent, tracking progress through the task processing workflow.
    
    Attributes:
        input: The original user input
        task_metadata: Metadata about the main task
        task_judgment: Current judgment of the task
        task_judgment_retry: Retry tracking for task judgment
        subtask_metadata: Metadata about the subtasks
        subtask_judgment: Current judgment of the subtasks
        subtask_judgment_retry: Retry tracking for subtask judgment
        user_wants_subtasks: Whether the user wants to break down the task into subtasks
        user_accepted_subtasks: Persistent flag indicating whether the user has accepted the subtasks.
                               This is preserved across iterations and can be used to influence the
                               workflow. It is updated based on the LLM's interpretation in
                               subtask_metadata.user_accepted_subtasks.
        last_user_message: The last message shown to the user
        user_feedback: The user's most recent feedback
        task_creation_confirmed: Whether the task has been created
    """
    input: Optional[str] = None
    task_metadata: Optional[TaskMetadata] = None
    task_judgment: Optional[TaskJudgment] = None
    task_judgment_retry: Optional[UserFeedbackRetry] = None
    subtask_metadata: Optional[SubtaskMetadata] = None
    subtask_judgment: Optional[SubtaskJudgment] = None
    subtask_judgment_retry: Optional[UserFeedbackRetry] = None
    user_wants_subtasks: Optional[bool] = None
    user_accepted_subtasks: Optional[bool] = None
    last_user_message: Optional[str] = None
    user_feedback: Optional[str] = None
    task_creation_confirmed: bool = False
