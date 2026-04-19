"""
数据模型定义

干净的 State 定义，仅保留核心字段
"""

from typing import TypedDict, Annotated, Optional, Dict, Any, List
from langchain_core.messages import BaseMessage


def add_messages(left: list, right: list) -> list:
    return left + right


def merge_rag_docs(left: list, right: list) -> list:
    return left + right


def merge_tool_results(left: list, right: list) -> list:
    return left + right


def merge_slots(left: dict, right: dict) -> dict:
    return {**left, **right}


def merge_context_entity(left: dict, right: dict) -> dict:
    return {**left, **right}


def replace_session_status(left: str, right: str) -> str:
    return right


class DispatchResult(TypedDict):
    need_rag: bool
    need_tool: bool
    need_clarify: bool
    need_transfer: bool
    clarify_prompt: Optional[str]
    reason: str
    tool_call: Optional[Dict[str, Any]]


class AgentState(TypedDict):
    """Agent 状态类型定义 - 干净版本"""

    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str

    slots: Annotated[Dict[str, Any], "merge_slots"]
    context_entity: Annotated[Dict[str, Any], "merge_context_entity"]
    session_status: Annotated[str, "replace_session_status"]
    intent: Optional[str]
    turn_count: int

    rag_docs: Annotated[List[str], merge_rag_docs]
    tool_results: Annotated[List[Dict[str, Any]], merge_tool_results]