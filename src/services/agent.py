"""
Agent 服务模块

基于 LangGraph StateGraph 的 ReAct 代理

企业级客服流程：
1. intent_node - 意图识别
2. rag_node - 检索 FAQ
3. agent_node - 思考 & 决定 tool_call
4. tools_node - 执行工具
5. conditional edge - 循环 ReAct

工作流程：
- intent_node → router
- router → rag_node（闲聊/咨询）
- router → agent_node（订单/物流/用户信息）
- router → 直接转人工（投诉/敏感/复杂）
- agent_node → tools_node（有tool_calls）→ agent_node
- agent_node → 结束（无tool_calls）
- tools_node → 回答生成
"""

import os
import logging
from typing import Generator

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from src.config.settings import config
from src.config.logger import get_logger
from src.models.types import AgentState

from src.services.tools import get_all_tools
from src.services.memory import get_memory
from src.services.intent import Intent, recognize_intent
from src.services.rag import get_rag
from src.services.validator import (
    InputValidator,
    ExceptionHandler,
    ValidationError,
)

logger = get_logger(__name__)

# 初始化 LangSmith 追踪（可选，用于调试）
if config.langsmith.api_key:
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith.api_key
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith.project_name
    logger.info(f"LangSmith tracing enabled: project={config.langsmith.project_name}")


# ==================== 工具函数映射 ====================

from src.services.tools import (
    query_order,
    query_logistics,
    query_user_info,
    transfer_to_human,
)

TOOL_MAP = {
    "query_order": query_order,
    "query_logistics": query_logistics,
    "query_user_info": query_user_info,
    "transfer_to_human": transfer_to_human,
}


# ==================== LLM 工具绑定缓存 ====================

from src.services.llm import get_llm_with_tools


# ==================== 节点函数 ====================


def intent_node(state: AgentState) -> AgentState:
    """
    意图识别节点
    识别用户意图，设置 state.intent
    """
    messages = state["messages"]
    last_message = messages[-1].content

    intent = recognize_intent(last_message)
    logger.info(f"Intent: {intent.value}")

    return {"intent": intent}


def rag_node(state: AgentState) -> AgentState:
    """
    RAG 检索节点（✅ 已修复 20015）
    """
    messages = state["messages"]
    user_query = messages[-1].content  # 真实用户问题

    # RAG 检索
    try:
        rag = get_rag()
        docs = rag.similarity_search(user_query, k=2)
        rag_context = "\n".join([d.page_content for d in docs]) if docs else ""
    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        rag_context = ""

    # 构建回答
    system_prompt = f"""你是一个智能客服助手。请根据以下知识库内容回答用户问题。

知识库：
{rag_context}

要求：
- 只根据知识库内容回答，不要编造
- 如果知识库没有相关信息，请如实告知用户"""

    llm_with_tools = get_llm_with_tools()

    # ✅ 【修复】最后一条 100% 是真实用户消息
    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    response = llm_with_tools.invoke(llm_messages)
    return {"messages": [response]}


def agent_node(state: AgentState) -> AgentState:
    """
    Agent 节点（✅ 已修复 ToolMessage + 20015）
    """
    messages = state["messages"]
    user_query = messages[-1].content
    llm_with_tools = get_llm_with_tools()

    # 工具返回结果 → 生成回答
    if isinstance(messages[-1], ToolMessage):
        system_prompt = """你是智能客服助手，根据工具结果自然回答用户。"""
        # ✅ 修复：必须带上用户原始问题
        llm_messages = [
            SystemMessage(content=system_prompt),
            *messages,
            HumanMessage(content=user_query),
        ]
        response = llm_with_tools.invoke(llm_messages)
        return {"messages": [response]}

    # 正常对话
    system_prompt = """你是一个智能客服助手。请严格遵守以下规则：
## 核心原则
1. 只能回答与客服相关的问题（商品、订单、物流、支付、售后等）
2. 如果不知道答案，直接说不知道，不要编造虚假信息
3. 保持专业、友好、简洁的回答风格

## 工具使用
- 订单查询关键词：订单号、查订单、订单状态
- 物流查询关键词：物流、快递、发货、到哪了
- 转人工关键词：转人工、投诉、找客服
"""

    # ✅ 修复：只保留 system + 用户真实问题
    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    response = llm_with_tools.invoke(llm_messages)
    return {"messages": [response]}


def tools_node(state: AgentState) -> AgentState:
    """
    工具执行节点
    """
    messages = state["messages"]
    last_msg = messages[-1]

    tool_calls = getattr(last_msg, "tool_calls", None)
    if not tool_calls:
        return {"messages": [AIMessage(content="无工具调用")]}

    tool_call = tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_call_id = tool_call.get("id", "")
    logger.info(f"Tool call: {tool_name}, args: {tool_args}")

    tool_func = TOOL_MAP.get(tool_name)
    if tool_func:
        tool_result = tool_func.invoke(tool_args)
        tool_response = f"为您查询到：\n{tool_result}"
        return {
            "messages": [
                ToolMessage(
                    content=tool_response,
                    tool_call_id=tool_call_id,
                )
            ]
        }
    else:
        return {
            "messages": [
                ToolMessage(
                    content=f"未知工具: {tool_name}",
                    tool_call_id=tool_call_id,
                )
            ]
        }


# ==================== 构建 Graph ====================


def create_agent_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("rag", rag_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("intent")

    def router(state: AgentState) -> str:
        intent = state.get("intent")
        if not intent:
            return "rag"

        if intent == Intent.TRANSFER_HUMAN:
            return "end"

        if intent in [Intent.ORDER_QUERY, Intent.LOGISTICS_QUERY, Intent.USER_QUERY]:
            return "agent"

        return "rag"

    graph.add_conditional_edges(
        "intent",
        router,
        {
            "rag": "rag",
            "agent": "agent",
            "end": END,
        },
    )

    def should_call_tool(state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"
        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "end"

    graph.add_conditional_edges(
        "agent",
        should_call_tool,
        {
            "tools": "tools",
            "end": END,
        },
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


# ==================== Agent 实例 ====================

_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
        logger.info("LangGraph Agent created")
    return _agent_graph


def run_agent(session_id: str, user_input: str) -> Generator[str, None, None]:
    """
    运行 Agent 处理用户输入

    Args:
        session_id: 会话 ID
        user_input: 用户输入

    Yields:
        流式输出的文本片段

    Raises:
        ValidationError: 输入验证失败
    """
    validation_result = InputValidator.validate_message(user_input)
    if not validation_result.is_valid:
        raise ValidationError(validation_result.error_message)

    validation_result = InputValidator.validate_session_id(session_id)
    if not validation_result.is_valid:
        raise ValidationError(validation_result.error_message)

    memory = get_memory(session_id)
    history = memory.get_messages()

    initial_state = {
        "messages": history + [HumanMessage(content=user_input)],
        "session_id": session_id,
        "intent": None,
    }

    full_response = ""
    for event in get_agent_graph().stream(initial_state):
        if "intent" in event:
            continue

        for node_name in ["rag", "agent"]:
            if node_name in event:
                messages = event[node_name].get("messages", [])
                if messages:
                    msg = messages[-1]
                    if msg.content:
                        new_content = msg.content[len(full_response) :]
                        full_response += new_content
                        if new_content:
                            yield new_content

        if "tools" in event:
            pass

    memory.add_user_message(user_input)
    memory.add_ai_message(full_response)
    logger.info(f"Agent response completed for session {session_id}")
