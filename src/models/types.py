"""
数据模型定义

包含 Agent 状态、消息等数据类型定义
"""

from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Agent 状态类型定义"""

    messages: Annotated[list[BaseMessage], "add_messages"]
    session_id: str
    intent: Optional[str]
