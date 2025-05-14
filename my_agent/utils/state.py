from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence
from pydantic import BaseModel

class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
