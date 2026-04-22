"""
对话记忆服务模块

纯内存存储，进程级生命周期
"""

import json
from typing import Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.config.logger import get_logger

logger = get_logger(__name__)

_memory_store: dict[str, list[BaseMessage]] = {}
_intent_store: dict[str, str] = {}
_slots_store: dict[str, dict] = {}
_task_state_store: dict[str, str] = {}
_turn_count_store: dict[str, int] = {}
_context_entity_store: dict[str, dict] = {}
_session_status_store: dict[str, str] = {}


class ChatMemory:
    """对话记忆类"""

    def __init__(self, session_id: str):
        self.session_id = session_id

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self.add_message(AIMessage(content=message))

    def add_message(self, msg: BaseMessage) -> None:
        """添加消息"""
        if self.session_id not in _memory_store:
            _memory_store[self.session_id] = []
        _memory_store[self.session_id].append(msg)

    def get_messages(self) -> list[BaseMessage]:
        """获取对话历史"""
        return _memory_store.get(self.session_id, [])

    def clear(self) -> None:
        """清空对话历史"""
        if self.session_id in _memory_store:
            del _memory_store[self.session_id]
        if self.session_id in _intent_store:
            del _intent_store[self.session_id]
        if self.session_id in _slots_store:
            del _slots_store[self.session_id]
        if self.session_id in _task_state_store:
            del _task_state_store[self.session_id]
        if self.session_id in _turn_count_store:
            del _turn_count_store[self.session_id]
        if self.session_id in _context_entity_store:
            del _context_entity_store[self.session_id]
        if self.session_id in _session_status_store:
            del _session_status_store[self.session_id]

    def set_intent(self, intent: Optional[str]) -> None:
        """保存意图"""
        _intent_store[self.session_id] = intent or ""

    def get_intent(self) -> Optional[str]:
        """获取上次保存的意图"""
        return _intent_store.get(self.session_id)

    def add_slots(self, slots: dict) -> None:
        """保存槽位"""
        if self.session_id not in _slots_store:
            _slots_store[self.session_id] = {}
        _slots_store[self.session_id].update(slots)

    def get_slots(self) -> dict:
        """获取槽位"""
        return _slots_store.get(self.session_id, {})

    def set_task_state(self, state: str) -> None:
        """保存任务状态"""
        _task_state_store[self.session_id] = state

    def get_task_state(self) -> str:
        """获取任务状态"""
        return _task_state_store.get(self.session_id, "pending")

    def increment_turn(self) -> int:
        """增加轮次"""
        count = _turn_count_store.get(self.session_id, 0) + 1
        _turn_count_store[self.session_id] = count
        return count

    def get_turn_count(self) -> int:
        """获取轮次"""
        return _turn_count_store.get(self.session_id, 0)

    def add_context_entity(self, entity: dict) -> None:
        """保存上下文实体"""
        if self.session_id not in _context_entity_store:
            _context_entity_store[self.session_id] = {}
        _context_entity_store[self.session_id].update(entity)

    def get_context_entity(self) -> dict:
        """获取上下文实体"""
        return _context_entity_store.get(self.session_id, {})

    def set_session_status(self, status: str) -> None:
        """保存会话状态"""
        valid_statuses = ["idle", "waiting"]
        if status not in valid_statuses:
            logger.warning(f"Invalid session_status: {status}, default to idle")
            status = "idle"
        _session_status_store[self.session_id] = status

    def get_session_status(self) -> str:
        """获取会话状态"""
        return _session_status_store.get(self.session_id, "idle")

    def save_session_state(self, state: dict) -> None:
        """保存完整会话状态（用于打断恢复）"""
        _slots_store[self.session_id] = state.get("slots", {})
        _context_entity_store[self.session_id] = state.get("context_entity", {})
        _session_status_store[self.session_id] = state.get("session_status", "idle")
        _intent_store[self.session_id] = state.get("intent", "")

    def restore_session_state(self) -> dict:
        """恢复会话状态"""
        return {
            "slots": _slots_store.get(self.session_id, {}),
            "context_entity": _context_entity_store.get(self.session_id, {}),
            "session_status": _session_status_store.get(self.session_id, "idle"),
            "intent": _intent_store.get(self.session_id, ""),
        }


def get_memory(session_id: str) -> ChatMemory:
    """获取对话记忆实例"""
    return ChatMemory(session_id)


def clear_memory(session_id: str) -> None:
    """清空指定会话的记忆"""
    memory = get_memory(session_id)
    memory.clear()