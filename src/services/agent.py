"""
Agent 服务模块

基于 LangChain create_agent 的 ReAct 代理
"""

import os
import logging
from typing import Generator, Optional

from langchain_core.messages import HumanMessage

from src.config.settings import config
from src.config.logger import get_logger
from src.services.llm import get_llm
from src.services.tools import get_all_tools
from src.services.rag import get_rag
from src.services.memory import get_memory
from src.services.intent import recognize_intent, Intent

logger = get_logger(__name__)

# 初始化 LangSmith 追踪
if config.langsmith.api_key:
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith.api_key
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith.project_name
    logger.info(f"LangSmith tracing enabled: project={config.langsmith.project_name}")

SYSTEM_PROMPT = """你是一个智能客服助手。请严格遵守以下规则：

## 核心原则
1. 只能回答与客服相关的问题（商品、订单、物流、支付、售后等），不要回答无关问题
2. 如果不知道答案，直接说不知道，不要编造虚假信息
3. 保持专业、友好、简洁的回答风格

## 意图识别规则
- 用户提到"订单"、"订单号"、"查订单" → 使用 query_order 工具
- 用户提到"物流"、"快递"、"发货"、"到哪了" → 使用 query_logistics 工具
- 用户要求"转人工"、"找客服"、"投诉" → 使用 transfer_to_human 工具
- 用户询问商品、售后、支付等常见问题 → 使用知识库（RAG）回答

## 工具使用时机
- 当用户明确提到订单号时，提取订单号并调用 query_order
- 当用户询问物流状态时，提取订单号并调用 query_logistics
- 当用户明确要求转人工时，调用 transfer_to_human
- 当不确定时，优先使用知识库回答

## 回答要求
- 首次回答应简洁明了
- 如需调用工具，明确告诉用户正在查询
- 回答结束时，可适当引导用户下一步需求
"""

_agent_instance: Optional["Agent"] = None


class Agent:
    """Agent 代理类"""

    def __init__(self):
        self.llm = get_llm()
        self.tools = get_all_tools()
        self._agent_graph = self._create_agent()

    def _create_agent(self):
        """创建 Agent 图"""
        from langchain.agents import create_agent

        agent = create_agent(
            self.llm,
            self.tools,
            system_prompt=SYSTEM_PROMPT,
        )
        logger.info("Agent graph created")
        return agent

    def run(self, session_id: str, user_input: str) -> Generator[str, None, None]:
        """
        执行 Agent 处理用户输入，支持流式输出

        Args:
            session_id: 会话 ID
            user_input: 用户输入

        Yields:
            生成的文本片段
        """
        memory = get_memory(session_id)
        history = memory.get_messages()

        # 意图识别
        intent = recognize_intent(user_input)
        logger.info(f"Recognized intent: {intent.value}")

        # RAG 检索
        try:
            rag = get_rag()
            docs = rag.similarity_search(user_input, k=3)
            context = "\n".join([d.page_content for d in docs]) if docs else ""
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            context = ""

        # 增强输入（添加意图信息）
        intent_hint = (
            f"\n\n[意图识别: {intent.value}]" if intent.value != "chat" else ""
        )
        if context:
            enhanced_input = f"{user_input}{intent_hint}\n\n相关知识：{context}"
        else:
            enhanced_input = f"{user_input}{intent_hint}"

        messages = list(history) + [HumanMessage(content=enhanced_input)]

        # 流式执行
        full_response = ""
        for chunk in self._agent_graph.stream({"messages": messages}):
            if "messages" in chunk:
                content = chunk["messages"][-1].content
            elif "model" in chunk and "messages" in chunk["model"]:
                content = chunk["model"]["messages"][-1].content
            else:
                continue

            if content:
                new_content = content[len(full_response) :]
                full_response = content
                yield new_content

        # 保存到记忆
        memory.add_user_message(user_input)
        memory.add_ai_message(full_response)

        logger.info(f"Agent response completed for session {session_id}")


def get_agent() -> Agent:
    """获取 Agent 实例"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = Agent()
    return _agent_instance


def run_agent(session_id: str, user_input: str) -> Generator[str, None, None]:
    """
    运行 Agent 的便捷函数

    Args:
        session_id: 会话 ID
        user_input: 用户输入

    Yields:
        生成的文本片段
    """
    agent = get_agent()
    yield from agent.run(session_id, user_input)
