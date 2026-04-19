"""
对话记忆服务模块

支持 Redis 存储和内存存储（降级方案）

序列化说明：
- 使用 JSON 序列化（安全可靠）
- 使用 Base64 编码存储二进制数据
"""

import json
import logging
from typing import Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from src.config.settings import config
from src.config.logger import get_logger

logger = get_logger(__name__)

_redis_client = None
_use_redis = False
_memory_store: dict[str, list[BaseMessage]] = {}
_intent_store: dict[str, str] = {}
_slots_store: dict[str, dict] = {}
_task_state_store: dict[str, str] = {}
_turn_count_store: dict[str, int] = {}
_context_entity_store: dict[str, dict] = {}
_session_status_store: dict[str, str] = {}


def _get_redis_client():
    """获取 Redis 客户端（带降级处理）"""
    global _redis_client, _use_redis

    if _redis_client is not None:
        return _redis_client

    try:
        import redis

        redis_config = config.redis
        _redis_client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=5,
        )
        _redis_client.ping()
        _use_redis = True
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed, using in-memory storage: {e}")
        _use_redis = False

    return _redis_client


class ChatMemory:
    """对话记忆类"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._key = f"chat_memory:{session_id}"
        self._intent_key = f"chat_intent:{session_id}"
        self._slots_key = f"chat_slots:{session_id}"
        self._task_key = f"chat_task:{session_id}"
        self._turn_key = f"chat_turns:{session_id}"
        self._context_entity_key = f"chat_context_entity:{session_id}"
        self._session_status_key = f"chat_session_status:{session_id}"

    def _get_client(self):
        return _get_redis_client()

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self.add_message(AIMessage(content=message))

    def add_message(self, msg: BaseMessage) -> None:
        """添加消息"""
        if _use_redis:
            client = self._get_client()
            if client:
                msgs = self.get_messages()
                msgs.append(msg)
                serialized = self._serialize_messages(msgs)
                client.set(self._key, serialized)
        else:
            if self.session_id not in _memory_store:
                _memory_store[self.session_id] = []
            _memory_store[self.session_id].append(msg)

    def get_messages(self) -> list[BaseMessage]:
        """获取对话历史"""
        if _use_redis:
            client = self._get_client()
            if client:
                data = client.get(self._key)
                if not data:
                    return []
                try:
                    return self._deserialize_messages(data)
                except Exception as e:
                    logger.warning(f"Failed to deserialize messages: {e}")
                    return []
        return _memory_store.get(self.session_id, [])

    def load_memory_variables(self) -> dict:
        """获取记忆变量（兼容 LangChain）"""
        return {"history": self.get_messages()}

    def clear(self) -> None:
        """清空对话历史"""
        if _use_redis:
            client = self._get_client()
            if client:
                client.delete(self._key)
                client.delete(self._intent_key)
                client.delete(self._slots_key)
                client.delete(self._task_key)
                client.delete(self._turn_key)
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

    def set_intent(self, intent: Optional[str]) -> None:
        """保存意图"""
        if _use_redis:
            client = self._get_client()
            if client:
                client.set(self._intent_key, intent or "")
        else:
            if not hasattr(self, "_intent_store"):
                global _intent_store
                _intent_store = {}
            _intent_store[self.session_id] = intent or ""

    def get_intent(self) -> Optional[str]:
        """获取上次保存的意图"""
        if _use_redis:
            client = self._get_client()
            if client:
                intent = client.get(self._intent_key)
                if intent:
                    return intent.decode("utf-8")
        else:
            global _intent_store
            return _intent_store.get(self.session_id)
        return None

    def add_slots(self, slots: dict) -> None:
        """保存槽位"""
        if _use_redis:
            client = self._get_client()
            if client:
                existing = self.get_slots()
                merged = {**existing, **slots}
                client.set(self._slots_key, json.dumps(merged, ensure_ascii=False))
        else:
            if self.session_id not in _slots_store:
                _slots_store[self.session_id] = {}
            _slots_store[self.session_id].update(slots)

    def get_slots(self) -> dict:
        """获取槽位"""
        if _use_redis:
            client = self._get_client()
            if client:
                data = client.get(self._slots_key)
                if data:
                    return json.loads(data.decode("utf-8"))
        return _slots_store.get(self.session_id, {})

    def set_task_state(self, state: str) -> None:
        """保存任务状态"""
        if _use_redis:
            client = self._get_client()
            if client:
                client.set(self._task_key, state)
        else:
            _task_state_store[self.session_id] = state

    def get_task_state(self) -> str:
        """获取任务状态"""
        if _use_redis:
            client = self._get_client()
            if client:
                data = client.get(self._task_key)
                if data:
                    return data.decode("utf-8")
        return _task_state_store.get(self.session_id, "pending")

    def increment_turn(self) -> int:
        """增加轮次"""
        if _use_redis:
            client = self._get_client()
            if client:
                count = client.incr(self._turn_key)
                if count is None:
                    return 1
                return int(count)
        else:
            count = _turn_count_store.get(self.session_id, 0) + 1
            _turn_count_store[self.session_id] = count
            return count

    def get_turn_count(self) -> int:
        """获取轮次"""
        if _use_redis:
            client = self._get_client()
            if client:
                data = client.get(self._turn_key)
                if data:
                    return int(data)
        return _turn_count_store.get(self.session_id, 0)

    def add_context_entity(self, entity: dict) -> None:
        """保存上下文实体"""
        if _use_redis:
            client = self._get_client()
            if client:
                existing = self.get_context_entity()
                merged = {**existing, **entity}
                client.set(
                    self._context_entity_key, json.dumps(merged, ensure_ascii=False)
                )
        else:
            if self.session_id not in _context_entity_store:
                _context_entity_store[self.session_id] = {}
            _context_entity_store[self.session_id].update(entity)

    def get_context_entity(self) -> dict:
        """获取上下文实体"""
        if _use_redis:
            client = self._get_client()
            if client:
                data = client.get(self._context_entity_key)
                if data:
                    return json.loads(data.decode("utf-8"))
        return _context_entity_store.get(self.session_id, {})

    def set_session_status(self, status: str) -> None:
        """保存会话状态"""
        valid_statuses = ["idle", "waiting_slot", "tool_running", "transfering"]
        if status not in valid_statuses:
            logger.warning(f"Invalid session_status: {status}, default to idle")
            status = "idle"

        if _use_redis:
            client = self._get_client()
            if client:
                client.set(self._session_status_key, status)
        else:
            _session_status_store[self.session_id] = status

    def get_session_status(self) -> str:
        """获取会话状态"""
        if _use_redis:
            client = self._get_client()
            if client:
                data = client.get(self._session_status_key)
                if data:
                    return data.decode("utf-8")
        return _session_status_store.get(self.session_id, "idle")

    def save_session_state(self, state: dict) -> None:
        """保存完整会话状态（用于打断恢复）"""
        if _use_redis:
            client = self._get_client()
            if client:
                full_state = {
                    "slots": json.dumps(state.get("slots", {})),
                    "context_entity": json.dumps(state.get("context_entity", {})),
                    "session_status": state.get("session_status", "idle"),
                    "intent": state.get("intent", ""),
                }
                client.hset(f"chat_session:{self.session_id}", mapping=full_state)
        else:
            _slots_store[self.session_id] = state.get("slots", {})
            _context_entity_store[self.session_id] = state.get("context_entity", {})
            _session_status_store[self.session_id] = state.get("session_status", "idle")
            _intent_store[self.session_id] = state.get("intent", "")

    def restore_session_state(self) -> dict:
        """恢复会话状态"""
        if _use_redis:
            client = self._get_client()
            if client:
                data = client.hgetall(f"chat_session:{self.session_id}")
                if data:
                    return {
                        "slots": json.loads(data.get(b"slots", b"{}")),
                        "context_entity": json.loads(
                            data.get(b"context_entity", b"{}")
                        ),
                        "session_status": data.get(b"session_status", b"idle").decode(
                            "utf-8"
                        ),
                        "intent": data.get(b"intent", b"").decode("utf-8"),
                    }
        return {
            "slots": _slots_store.get(self.session_id, {}),
            "context_entity": _context_entity_store.get(self.session_id, {}),
            "session_status": _session_status_store.get(self.session_id, "idle"),
            "intent": _intent_store.get(self.session_id, ""),
        }

    def _serialize_messages(self, messages: list[BaseMessage]) -> str:
        """序列化消息为 JSON 字符串"""
        data = []
        for msg in messages:
            msg_dict = {
                "type": msg.type,
                "content": msg.content,
            }
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            data.append(msg_dict)
        return json.dumps(data, ensure_ascii=False)

    def _deserialize_messages(self, data: bytes) -> list[BaseMessage]:
        """从 JSON 反序列化消息"""
        if isinstance(data, str):
            data = data.encode("utf-8")

        data_list = json.loads(data.decode("utf-8"))
        messages = []

        for msg_dict in data_list:
            msg_type = msg_dict.get("type", "human")
            content = msg_dict.get("content", "")

            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                msg = AIMessage(content=content)
                if msg_dict.get("tool_calls"):
                    msg.tool_calls = msg_dict["tool_calls"]
                messages.append(msg)
            elif msg_type == "tool":
                msg = ToolMessage(
                    content=content, tool_call_id=msg_dict.get("tool_call_id", "")
                )
                messages.append(msg)
            else:
                messages.append(HumanMessage(content=content))

        return messages


def get_memory(session_id: str) -> ChatMemory:
    """
    获取对话记忆实例

    Args:
        session_id: 会话 ID

    Returns:
        ChatMemory 实例
    """
    return ChatMemory(session_id)


def clear_memory(session_id: str) -> None:
    """
    清空指定会话的记忆

    Args:
        session_id: 会话 ID
    """
    memory = get_memory(session_id)
    memory.clear()
