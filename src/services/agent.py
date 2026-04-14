"""
Agent 服务模块

基于 LangGraph StateGraph 的 ReAct 代理
"""

import os
import logging
from typing import Generator, TypedDict, List
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.config.settings import config
from src.config.logger import get_logger
from src.services.tools import (
    get_all_tools,
    query_order,
    query_logistics,
    query_user_info,
    transfer_to_human,
)
from src.services.rag import get_rag
from src.services.memory import get_memory
from src.services.intent import recognize_intent
from src.services.recall import get_recall_context

logger = get_logger(__name__)

# 初始化 LangSmith 追踪
if config.langsmith.api_key:
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith.api_key
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith.project_name
    logger.info(f"LangSmith tracing enabled: project={config.langsmith.project_name}")


# ==================== LangGraph 状态定义 ====================


class AgentState(TypedDict):
    """Agent 状态"""

    messages: list[BaseMessage]
    session_id: str


# ==================== 工具函数映射 ====================

TOOL_MAP = {
    "query_order": query_order,
    "query_logistics": query_logistics,
    "query_user_info": query_user_info,
    "transfer_to_human": transfer_to_human,
}


# ==================== LLM 创建 ====================

_llm_instance = None


def create_llm():
    """创建 LLM 实例（单例模式）"""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if config.llm_provider == "siliconflow":
        from langchain_openai import ChatOpenAI

        _llm_instance = ChatOpenAI(
            model=config.siliconflow.model,
            base_url=config.siliconflow.base_url,
            api_key=config.siliconflow.api_key,
            temperature=config.llm.temperature,
        )
    else:
        from langchain_ollama import ChatOllama

        _llm_instance = ChatOllama(
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
        )

    return _llm_instance


# ==================== LLM 工具绑定缓存 ====================

_llm_with_tools = None


def get_llm_with_tools():
    """获取绑定工具后的 LLM（缓存）"""
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = get_llm()
        tools = get_all_tools()
        _llm_with_tools = llm.bind_tools(tools)
    return _llm_with_tools


# ==================== 节点函数 ====================


def chat_node(state: AgentState) -> AgentState:
    """
    聊天节点 - 使用 LLM 生成回复
    """
    messages = state["messages"]
    last_message = messages[-1].content
    session_id = state["session_id"]

    # 并行执行意图识别、召回和 RAG
    with ThreadPoolExecutor(max_workers=3) as executor:
        intent_future = executor.submit(recognize_intent, last_message)
        recall_future = executor.submit(get_recall_context, last_message)
        rag_future = executor.submit(
            lambda: get_rag().similarity_search(last_message, k=3)
        )

        intent = intent_future.result().value
        logger.info(f"Intent: {intent}")
        recall_context = recall_future.result()
        try:
            docs = rag_future.result()
            rag_context = "\n".join([d.page_content for d in docs]) if docs else ""
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            rag_context = ""

    # 合并上下文
    context = f"{recall_context}\n\n{rag_context}" if recall_context else rag_context

    # 构建 prompt
    system_prompt = """你是一个智能客服助手。请严格遵守以下规则：

## 核心原则
1. 只能回答与客服相关的问题（商品、订单、物流、支付、售后等），不要回答无关问题
2. 如果不知道答案，直接说不知道，不要编造虚假信息
3. 保持专业、友好、简洁的回答风格

## 工具使用
- 订单查询关键词：订单号、查订单、订单状态
- 物流查询关键词：物流、快递、发货、到哪了
- 转人工关键词：转人工、投诉、找客服

## 回答要求
- 如需查询订单或物流，请明确告诉用户正在查询，并提供简短的用户友好说明
- 即使调用工具，也要先给用户一个回复（如"正在为您查询..."）
- 回答结束时，可适当引导用户下一步需求
"""

    if context:
        system_prompt += f"\n\n相关知识：{context}"

    # 获取绑好工具的 LLM
    llm_with_tools = get_llm_with_tools()

    # 调用 LLM
    response = llm_with_tools.invoke(
        [
            {"role": "system", "content": system_prompt},
            *[
                {
                    "role": "human" if isinstance(m, HumanMessage) else "assistant",
                    "content": m.content,
                }
                for m in messages
            ],
        ]
    )

    # 检查是否有工具调用
    if response.tool_calls:
        tool_name = response.tool_calls[0]["name"]
        tool_args = response.tool_calls[0]["args"]
        logger.info(f"Tool call: {tool_name}, args: {tool_args}")

        # 执行工具
        tool_func = TOOL_MAP.get(tool_name)
        if tool_func:
            tool_result = tool_func.invoke(tool_args)

            # 直接使用工具返回的结果（已经是格式化好的字符串）
            final_content = f"为您查询到：\n{tool_result}"

            return {"messages": [AIMessage(content=final_content)]}
        else:
            return {"messages": [AIMessage(content=f"未知工具: {tool_name}")]}
    else:
        return {"messages": [AIMessage(content=response.content)]}


# ==================== 构建 Graph ====================


def create_agent_graph():
    """创建 LangGraph Agent"""
    from langgraph.graph import StateGraph, END

    # 创建图
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("chat", chat_node)

    # 设置入口
    graph.set_entry_point("chat")

    # 添加条件边
    graph.add_conditional_edges(
        "chat",
        lambda state: "continue" if state.get("messages", [])[-1].tool_calls else "end",
        {
            "continue": "chat",
            "end": END,
        },
    )

    return graph.compile()


# ==================== Agent 实例 ====================

_agent_graph = None


def get_agent_graph():
    """获取 Agent Graph 实例"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
        logger.info("LangGraph Agent created")
    return _agent_graph


def run_agent(session_id: str, user_input: str) -> Generator[str, None, None]:
    """
    运行 Agent 处理用户输入，支持流式输出
    """
    memory = get_memory(session_id)
    history = memory.get_messages()

    # 构建初始状态
    initial_state = {
        "messages": history + [HumanMessage(content=user_input)],
        "session_id": session_id,
    }

    # 执行图
    full_response = ""
    for event in get_agent_graph().stream(initial_state):
        if "chat" in event:
            messages = event["chat"].get("messages", [])
            if messages:
                content = messages[-1].content
                if content:
                    new_content = content[len(full_response) :]
                    full_response = content
                    yield new_content

    # 保存到记忆
    memory.add_user_message(user_input)
    memory.add_ai_message(full_response)
    logger.info(f"Agent response completed for session {session_id}")
