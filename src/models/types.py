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


class ToolResult:
    """工具执行结果"""

    def __init__(
        self, name: str, result: str, success: bool = True, error: Optional[str] = None
    ):
        self.name = name
        self.result = result
        self.success = success
        self.error = error

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "result": self.result,
            "success": self.success,
            "error": self.error,
        }


class SessionContext:
    """会话上下文"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: list[BaseMessage] = []
        self.metadata: dict = {}

    def add_user_message(self, content: str):
        from langchain_core.messages import HumanMessage

        self.history.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        from langchain_core.messages import AIMessage

        self.history.append(AIMessage(content=content))

    def get_history(self) -> list[BaseMessage]:
        return self.history

    def clear(self):
        self.history = []
        self.metadata = {}
