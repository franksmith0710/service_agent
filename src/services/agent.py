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
import json
import logging
from typing import Generator, Any, Dict

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
from src.services.intent import (
    Intent,
    recognize_intent,
    get_rag_filter,
    extract_order_id,
    extract_phone,
)
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


def transfer_node(state: AgentState) -> AgentState:
    """
    转人工节点
    """
    return {
        "messages": [
            AIMessage(
                content="已为您转接人工客服，请稍候...\n人工客服工作时间: 周一至周五 9:00-18:00\n客服热线: 400-990-5898"
            )
        ]
    }


def chat_node(state: AgentState) -> AgentState:
    """聊天节点 - 处理问候、闲聊、礼貌对话"""
    messages = state["messages"]
    user_input = messages[-1].content.strip().lower()

    greetings = {"你好", "您好", "hi", "hello", "在吗", "在么", "有人吗", "哈喽"}
    thanks = {"谢谢", "感谢", "谢了", "多谢"}
    goodbyes = {"再见", "拜拜", "bye", "88", "回头聊"}

    if any(word in user_input for word in greetings):
        content = "您好！我是智能客服，有什么可以帮您的吗？😊"
    elif any(word in user_input for word in thanks):
        content = "不客气！很高兴能帮到您~"
    elif any(word in user_input for word in goodbyes):
        content = "再见！感谢您的咨询，祝您生活愉快！👋"
    else:
        content = "您好！我可以帮您查询产品、售后、订单、物流等问题，请问需要什么帮助？"

    return {"messages": [AIMessage(content=content)]}


def intent_node(state: AgentState) -> AgentState:
    """
    意图识别节点
    识别用户意图，设置 state.intent 和 rag_filter
    """
    messages = state["messages"]
    last_message = messages[-1].content

    intent = recognize_intent(last_message)

    logger.info(f"Intent: {intent.value}")

    rag_filter = get_rag_filter(intent)
    logger.info(f"RAG filter: {rag_filter}")

    return {"intent": intent.value, "rag_filter": rag_filter}


def _rag_ask_back(intent: str) -> AgentState:
    """RAG 检索为空时的追问"""
    question_map = {
        "product": "您好！请问您想咨询哪款产品的具体型号或配置呢？",
        "pre_sales": "您好！请问您的预算和使用场景是什么呢？我可以为您推荐合适的机型。",
        "after_sales": "您好！请描述一下您遇到的故障现象，例如蓝屏、死机、无法开机等。",
    }
    fallback = "您好！请问您具体想咨询什么问题呢？"
    response_content = question_map.get(intent, fallback)

    return {"messages": [AIMessage(content=response_content)]}


def rag_node(state: AgentState) -> AgentState:
    """
    RAG 检索节点
    """
    messages = state["messages"]
    user_query = messages[-1].content
    intent = state.get("intent")

    rag_filter = state.get("rag_filter")
    logger.info(f"RAG filter: {rag_filter}")

    history_ai = []
    for msg in messages[:-1]:
        if isinstance(msg, AIMessage):
            history_ai.append(msg.content)
    last_ai_reply = history_ai[-1] if history_ai else ""

    PRODUCT_KEYWORDS = ["耀世", "蛟龙", "极光", "无界", "旷世", "翼龙", "深海泰坦"]
    has_specific_product = any(kw in user_query for kw in PRODUCT_KEYWORDS)

    if not has_specific_product:
        return _rag_ask_back(intent)

    try:
        rag = get_rag()
        docs = rag.similarity_search(user_query, k=2, filter=rag_filter)
    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        docs = []

    if not docs:
        return _rag_ask_back(intent)

    rag_context = "\n".join([d.page_content for d in docs])

    system_prompt = f"""你是一个智能客服助手。请根据以下知识库内容回答用户问题。

知识库：
{rag_context}

要求：
- 只根据知识库内容回答，不要编造
- 如果知识库没有相关信息，请如实告知用户"""

    if last_ai_reply:
        system_prompt += f"""

参考上一轮对话：
{last_ai_reply}

用户当前问题：{user_query}"""
    else:
        system_prompt += f"""

用户当前问题：{user_query}"""

    llm_with_tools = get_llm_with_tools()

    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    response = llm_with_tools.invoke(llm_messages)
    return {"messages": [response]}


def agent_node(state: AgentState) -> AgentState:
    """
    Agent 节点（ReAct 工具调用）
    """
    messages = state["messages"]
    last_msg = messages[-1]
    llm_with_tools = get_llm_with_tools()

    history_ai = []
    for msg in messages[:-1]:
        if isinstance(msg, AIMessage):
            history_ai.append(msg.content)
    last_ai_reply = history_ai[-1] if history_ai else ""

    if isinstance(last_msg, ToolMessage):
        system_prompt = """你是智能客服助手，根据工具结果自然回答用户。
要求：简洁、专业、直接回答用户问题。
"""
        context_msg = f"工具查询结果：{last_msg.content}"
        llm_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_msg),
        ]
        response = llm_with_tools.invoke(llm_messages)
        return {"messages": [response]}

    user_query = messages[-1].content
    intent = state.get("intent")

    order_id = extract_order_id(user_query)
    phone = extract_phone(user_query)

    if intent == "order" and not order_id and not phone:
        return {"messages": [AIMessage(content="请提供订单号或手机号以便查询订单")]}
    if intent == "logistics" and not order_id and not phone:
        return {"messages": [AIMessage(content="请提供订单号或手机号以便查询物流")]}
    if intent == "user" and not phone:
        return {"messages": [AIMessage(content="请提供手机号以便查询用户信息")]}

    system_prompt = """��是一个智能客服助手。根据用户问题，直接调用工具查询。
如果需要查询订单，请调用 query_order 工具。
如果需要查询物流，请调用 query_logistics 工具。
如果需要查询用户信息，请调用 query_user_info 工具。
不要引导用户，直接回答问题。"""

    if last_ai_reply:
        system_prompt += f"""

参考对话：
{last_ai_reply}

用户当前问题：{user_query}"""
    else:
        system_prompt += f"""

用户当前问题：{user_query}"""

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
    tool_args_raw = tool_call.get("args", {})
    tool_call_id = tool_call.get("id", "")

    if isinstance(tool_args_raw, str):
        tool_args = json.loads(tool_args_raw) if tool_args_raw else {}
    elif isinstance(tool_args_raw, dict):
        tool_args = tool_args_raw
    else:
        tool_args = {}

    logger.info(f"Tool call: {tool_name}, args: {tool_args}")

    tool_func = TOOL_MAP.get(tool_name)
    if tool_func:
        try:
            tool_result = tool_func.invoke(tool_args)
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            tool_result = f"工具执行失败: {str(e)}"
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
    graph.add_node("chat", chat_node)
    graph.add_node("rag", rag_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("transfer", transfer_node)

    graph.set_entry_point("intent")

    def router(state: AgentState) -> str:
        intent = state.get("intent")
        if not intent:
            return "rag"

        # 转人工 → 工具调用
        if intent == "transfer":
            return "agent"

        # 聊天节点（问候/闲聊）
        if intent == "chat":
            return "chat"

        # 产品/售前/售后咨询 → RAG（带 filter）
        if intent in ["product", "pre_sales", "after_sales"]:
            return "rag"

        # 订单、物流、用户 → 工具调用
        if intent in ["order", "logistics", "user"]:
            return "agent"

        return "chat"

    graph.add_conditional_edges(
        "intent",
        router,
        {
            "chat": "chat",
            "rag": "rag",
            "agent": "agent",
            "transfer": "transfer",
        },
    )

    graph.add_edge("transfer", END)
    graph.add_edge("chat", END)
    graph.add_edge("rag", END)

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


def run_agent(
    session_id: str, user_input: str, enable_stream: bool = True
) -> Generator[str, None, None]:
    """
    运行 Agent 处理用户输入

    Args:
        session_id: 会话 ID
        user_input: 用户输入
        enable_stream: 是否启用流式输出

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
    prev_intent = memory.get_intent()

    initial_state = {
        "messages": history + [HumanMessage(content=user_input)],
        "session_id": session_id,
        "intent": prev_intent,
    }

    llm_with_tools = get_llm_with_tools()
    full_response = ""
    response_done = False
    current_intent = None

    for event in get_agent_graph().stream(initial_state):
        if "intent" in event:
            current_intent = event["intent"].get("intent")
            continue

        if response_done:
            break

        if "tools" in event:
            continue

        for node_name in ["chat", "rag", "agent"]:
            if response_done:
                break

            if node_name in event:
                messages = event[node_name].get("messages", [])
                if messages:
                    msg = messages[-1]
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        continue

                    if msg.content:
                        content = msg.content
                        if enable_stream:
                            for char in content:
                                yield char
                        else:
                            yield content
                        full_response = content
                        response_done = True
                        break
                break

    memory.add_user_message(user_input)
    memory.add_ai_message(full_response)

    if current_intent:
        memory.set_intent(current_intent)

    logger.info(f"Agent response completed for session {session_id}")
