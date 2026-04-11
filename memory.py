from typing import Optional, List
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    messages_to_dict,
)


class ChatMemory:
    def __init__(self, return_messages: bool = True):
        self.return_messages = return_messages
        self.messages: List[BaseMessage] = []

    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))

    def get_messages(self) -> List[BaseMessage]:
        return self.messages

    def load_memory_variables(self) -> dict:
        return {"history": self.messages}

    def clear(self):
        self.messages = []


def create_chat_memory() -> ChatMemory:
    return ChatMemory()


_memory_store: dict[str, ChatMemory] = {}


def get_memory(session_id: str) -> ChatMemory:
    if session_id not in _memory_store:
        _memory_store[session_id] = create_chat_memory()
    return _memory_store[session_id]


def clear_memory(session_id: str):
    if session_id in _memory_store:
        _memory_store[session_id].clear()
