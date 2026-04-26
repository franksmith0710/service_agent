"""
数据模型定义

定义 Agent 状态类型和调度结果类型
"""

from typing import TypedDict, Annotated, Optional, Dict, Any, List
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


def add_messages(left: list, right: list) -> list:
    return left + right


def merge_rag_docs(left: list, right: list) -> list:
    return right  # 只保留最新一次检索结果


def merge_tool_results(left: list, right: list) -> list:
    return left + right  # 拼接，不覆盖！


def merge_list(left: list, right: list) -> list:
    return right  # 以本次最新队列为准，避免重复追加


def merge_slots(left: dict, right: dict) -> dict:
    return {**left, **right}


def merge_context_entity(left: dict, right: dict) -> dict:
    return {**left, **right}


def replace_session_status(left: str, right: str) -> str:
    return right


class DispatchResult(BaseModel):
    """调度决策结果"""

    need_rag: bool = False
    need_tool: bool = False
    need_clarify: bool = False
    tool_calls: List[Dict[str, Any]] = []
    clarify_prompt: str = ""


class AgentState(TypedDict):
    """Agent 状态类型定义"""

    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str

    slots: Annotated[Dict[str, Any], merge_slots]
    context_entity: Annotated[Dict[str, Any], merge_context_entity]
    session_status: Annotated[str, replace_session_status]
    intent: Optional[str]
    turn_count: int

    rag_docs: Annotated[List[str], merge_rag_docs]
    tool_results: Annotated[List[Dict[str, Any]], merge_tool_results]

    tool_calls: Annotated[List[Dict[str, Any]], merge_list]
    need_rag: bool
    need_tool: bool
    need_clarify: bool
    clarify_prompt: Optional[str]

    # ========== 工具循环控制字段 ==========
    tool_queue: Annotated[List[Dict[str, Any]], merge_list]
    tool_exec_count: int
    max_tool_limit: int

    # ========== 第三步新增：会话状态机 ==========
    step: str                         # 当前流程步骤：init → check_slots → clarify → tools → rag → summary → end
    next_step: Optional[str]          # 下一步路由