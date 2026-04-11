"""
对话记忆服务模块

支持 Redis 存储和内存存储（降级方案）
"""

import json
import logging
from typing import Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    messages_to_dict,
    messages_from_dict,
)

from src.config.settings import config
from src.config.logger import get_logger

logger = get_logger(__name__)

_redis_client = None
_use_redis = False
_memory_store: dict[str, list[BaseMessage]] = {}


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
            decode_responses=True,
            socket_connect_timeout=2,
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
                client.set(self._key, json.dumps(messages_to_dict(msgs)))
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
                    msg_dicts = json.loads(data)
                    return messages_from_dict(msg_dicts)
                except Exception:
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
        if self.session_id in _memory_store:
            del _memory_store[self.session_id]


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
