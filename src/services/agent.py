"""
Agent 服务模块

两层级 LLM 架构：
- 决策 LLM (run_agent 入口)：意图识别、路由决策
- Graph 执行层：RAG 检索 + 工具执行（0 LLM）
- 生成 LLM (run_agent 末尾)：整合所有原料生成回答
"""

import os
import json
import logging
from typing import Generator

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from src.config.settings import config
from src.config.logger import get_logger
from src.models.types import AgentState

from src.services.memory import get_memory
from src.services.intent import (
    llm_dispatch,
    extract_all_slots,
    resolve_coreference,
    update_context_entity,
    is_interrupt,
)
from src.services.rag import get_rag
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
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

# ==================== 工具失败固定策略 ====================

TOOL_ERROR_STRATEGY = {
    "query_order": "抱歉，订单查询服务暂时不可用，请稍后重试或拨打客服热线 400-990-5898",
    "query_logistics": "抱歉，物流查询服务暂时不可用，请稍后重试",
    "query_user_info": "抱歉，用户信息查询服务暂时不可用",
    "transfer_to_human": "抱歉，转接服务暂时不可用，请拨打客服热线 400-990-5898",
}

# ==================== 工具函数映射 ====================

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
    """简化后的意图节点 - 只做状态传递"""
    return {"session_status": state.get("session_status", "idle")}


def rag_node(state: AgentState) -> AgentState:
    """RAG 检索节点 - 只同步检索，返回原始文档"""
    messages = state["messages"]
    user_query = messages[-1].content

    logger.info(f"RAG search: {user_query[:30]}...")

    try:
        rag = get_rag()
        docs = rag.similarity_search(user_query, k=2)
        rag_docs = [d.page_content for d in docs]
    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        rag_docs = []

    logger.info(f"RAG docs: {len(rag_docs)}")

    return {"rag_docs": rag_docs, "session_status": "idle"}


MAX_TURNS = 8


def agent_node(state: AgentState) -> AgentState:
    """Agent 节点 - ReAct 循环继续"""
    messages = state.get("messages", [])
    if not messages:
        return {"session_status": "idle"}

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage):
        return {"session_status": "idle"}

    return {"session_status": "idle"}


def tools_node(state: AgentState) -> AgentState:
    """
    工具执行节点
    更新槽位、任务状态、last_tool
    """
    messages = state["messages"]
    last_msg = messages[-1]
    session_id = state.get("session_id", "")

    tool_calls = getattr(last_msg, "tool_calls", None)
    if not tool_calls:
        return {
            "messages": [AIMessage(content="无工具调用")],
            "session_status": "idle",
        }

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

    new_slots = {}
    if "order_id" in tool_args:
        new_slots["order_id"] = tool_args["order_id"]
    if "phone" in tool_args:
        new_slots["phone"] = tool_args["phone"]

    tool_func = TOOL_MAP.get(tool_name)
    if tool_func:
        try:
            tool_result = tool_func.invoke(tool_args)
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            tool_result = TOOL_ERROR_STRATEGY.get(tool_name, "服务暂时不可用")

        return {
            "messages": [
                AIMessage(content=tool_result)
            ],
            "tool_results": [{"name": tool_name, "result": tool_result}],
            "slots": new_slots,
            "session_status": "idle",
        }
    else:
        return {
            "messages": [
                AIMessage(content=f"未知工具: {tool_name}")
            ],
            "tool_results": [{"name": tool_name, "result": f"未知工具: {tool_name}"}],
            "session_status": "idle",
        }


# ==================== 构建 Graph ====================


def create_agent_graph():
    from langgraph.graph import StateGraph, END
    from langgraph.constants import Send

    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("rag", rag_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("intent")

    def simple_router(state: AgentState) -> str:
        """根据 intent 决定路由"""
        intent = state.get("intent", "chat")
        if intent == "product":
            return "rag"
        return "agent"

    graph.add_conditional_edges(
        "intent",
        simple_router,
        {
            "rag": "rag",
            "agent": "agent",
        },
    )

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
    核心：在 Graph 外完成决策，Graph 只做执行
    加载/保存 slots、context_entity、session_status、turn_count
    """
    memory = get_memory(session_id)
    history = memory.get_messages()
    prev_intent = memory.get_intent()
    saved_slots = memory.get_slots()
    saved_context_entity = memory.get_context_entity()
    prev_session_status = memory.get_session_status()
    prev_turn_count = memory.get_turn_count()

    if prev_turn_count >= MAX_TURNS:
        yield "对话已达最大轮次(8轮)，为保证服务质量已为您转接人工客服。\n人工客服工作时间: 周一至周五 9:00-18:00\n客服热线: 400-990-5898"
        memory.clear()
        memory.set_session_status("transfering")
        memory.increment_turn()
        return

    if is_interrupt(user_input):
        logger.info(f"User interrupted, saving session state")
        memory.set_session_status("idle")
        slots = extract_all_slots(user_input)
        merged_slots = {**saved_slots, **slots}
        new_context_entity = update_context_entity(merged_slots, saved_context_entity)
        memory.add_slots(merged_slots)
        memory.add_context_entity(new_context_entity)
        yield "好的，保存进度。"
        return

    if saved_context_entity:
        user_input = resolve_coreference(user_input, saved_context_entity)
        logger.info(f"Coreference resolved: {user_input}")

    raw_state = {
        "messages": history + [HumanMessage(content=user_input)],
        "session_id": session_id,
        "slots": saved_slots,
        "context_entity": saved_context_entity,
        "turn_count": prev_turn_count,
    }

    dispatch_result = llm_dispatch(raw_state)
    dispatch_reason = dispatch_result.reason
    need_rag = dispatch_result.need_rag
    need_tool = dispatch_result.need_tool
    need_clarify = dispatch_result.need_clarify
    need_transfer = dispatch_result.need_transfer
    tool_call = dispatch_result.tool_call

    logger.info(
        f"LLM dispatch: need_rag={need_rag}, need_tool={need_tool}, "
        f"tool_call={tool_call}, reason={dispatch_reason}"
    )

    intent = "product" if need_rag else "order"

    slots = extract_all_slots(user_input)
    if tool_call and tool_call.get("args"):
        slots = {**slots, **tool_call.get("args", {})}
    merged_slots = {**saved_slots, **slots}
    new_context_entity = update_context_entity(merged_slots, saved_context_entity)

    turn_count = prev_turn_count + 1

    messages = [HumanMessage(content=user_input)]
    if need_tool and tool_call:
        messages.append(
            AIMessage(content="", tool_calls=[tool_call])
        )

    initial_state = {
        "messages": messages,
        "session_id": session_id,
        "intent": intent,
        "rag_docs": [],
        "tool_results": [],
        "slots": merged_slots,
        "context_entity": new_context_entity,
        "session_status": "waiting_slot" if need_clarify else "idle",
        "turn_count": turn_count,
    }

    if need_clarify:
        clarify_prompt = dispatch_result.clarify_prompt or "请问您具体想咨询什么？"
        yield clarify_prompt
        memory.add_user_message(user_input)
        memory.set_intent(intent)
        memory.add_slots(merged_slots)
        memory.add_context_entity(new_context_entity)
        memory.set_session_status("waiting_slot")
        memory.increment_turn()
        return

    if need_transfer:
        yield "已为您转接人工客服，请稍候...\n人工客服工作时间: 周一至周五 9:00-18:00\n客服热线: 400-990-5898"
        memory.add_user_message(user_input)
        memory.set_intent("transfer")
        memory.set_session_status("transfering")
        memory.increment_turn()
        return

    if not need_rag and not need_tool:
        chat_greetings = {"你好", "您好", "hi", "hello", "在吗", "在么", "有人吗", "哈喽"}
        chat_thanks = {"谢谢", "感谢", "谢了", "多谢"}
        chat_goodbyes = {"再见", "拜拜", "bye", "88", "回头聊"}
        user_lower = user_input.strip().lower()
        if any(word in user_lower for word in chat_greetings):
            yield "您好！我是智能客服，有什么可以帮您的吗？😊"
        elif any(word in user_lower for word in chat_thanks):
            yield "不客气！很高兴能帮到您~"
        elif any(word in user_lower for word in chat_goodbyes):
            yield "再见！感谢您的咨询，祝您生活愉快！👋"
        else:
            yield "您好！我可以帮您查询产品、售后、订单、物流等问题，请问需要什么帮助？"
        memory.add_user_message(user_input)
        memory.add_ai_message("")
        memory.set_intent("chat")
        memory.increment_turn()
        return

    llm_with_tools = get_llm_with_tools()
    final_rag_docs = []
    final_tool_results = []
    final_session_status = "idle"
    current_intent = initial_state.get("intent")
    final_slots = initial_state.get("slots", {})
    final_context_entity = initial_state.get("context_entity", {})

    agent_graph = get_agent_graph()
    for event in agent_graph.stream(initial_state):
        if "intent" in event:
            final_session_status = event["intent"].get("session_status", "idle")
        if "rag" in event:
            final_rag_docs = event["rag"].get("rag_docs", [])
        if "tools" in event:
            tool_results = event["tools"].get("tool_results", [])
            final_tool_results.extend(tool_results)
            tools_slots = event["tools"].get("slots", {})
            if tools_slots:
                final_slots = {**final_slots, **tools_slots}

    llm_messages = [
        SystemMessage(content="你是机械革命官方客服，仅使用提供的资料回答，不编造。")
    ]

    if history:
        llm_messages.extend(history)

    rag_context = "\n".join(final_rag_docs) if need_rag else ""
    if rag_context:
        llm_messages.append(HumanMessage(content=f"参考资料：\n{rag_context}"))

    llm_messages.append(HumanMessage(content=user_input))

    final_response = llm_with_tools.invoke(llm_messages)
    final_response_text = final_response.content

    if enable_stream:
        for char in final_response_text:
            yield char
    else:
        yield final_response_text

    memory.add_user_message(user_input)
    memory.add_ai_message(final_response_text)

    if current_intent:
        memory.set_intent(current_intent)

    if final_slots:
        memory.add_slots(final_slots)

    if final_context_entity:
        memory.add_context_entity(final_context_entity)

    memory.set_session_status(final_session_status)
    memory.increment_turn()

    logger.info(
        f"Agent response completed for session {session_id}, "
        f"intent={current_intent}, session_status={final_session_status}, slots={final_slots}"
    )
