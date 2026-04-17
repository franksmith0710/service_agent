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
        if self.session_id in _memory_store:
            del _memory_store[self.session_id]
        if self.session_id in _intent_store:
            del _intent_store[self.session_id]

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
