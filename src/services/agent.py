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
    """RAG 检索节点 - 多路召回"""
    messages = state["messages"]
    user_query = messages[-1].content

    logger.info(f"RAG multi-search: {user_query[:30]}...")

    try:
        rag = get_rag()
        docs = rag.multi_search(user_query, k=2, vector_k=5, bm25_k=5)
        rag_docs = [d.page_content for d in docs]
    except Exception as e:
        logger.warning(f"RAG multi-search failed: {e}, fallback to vector search")
        try:
            docs = rag.similarity_search(user_query, k=2)
            rag_docs = [d.page_content for d in docs]
        except Exception as e2:
            logger.warning(f"RAG search also failed: {e2}")
            rag_docs = []

    logger.info(f"RAG docs: {len(rag_docs)}")

    return {"rag_docs": rag_docs, "session_status": "idle"}


MAX_TURNS = 8


def agent_node(state: AgentState) -> AgentState:
    """Agent 节点 - ReAct 循环继续 - 透传数据"""
    messages = state.get("messages", [])
    tool_results = state.get("tool_results", [])
    slots = state.get("slots", {})
    rag_docs = state.get("rag_docs", [])

    if not messages:
        return {"session_status": "idle"}

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return {"session_status": "idle"}  # 继续执行工具

    # 透传已有数据，不丢弃
    return {
        "session_status": "idle",
        "tool_results": tool_results,
        "slots": slots,
        "rag_docs": rag_docs,
    }


def tools_node(state: AgentState) -> AgentState:
    """
    工具执行节点
    从state中读取tool_name和tool_params执行
    """
    tool_name = state.get("tool_name", "")
    tool_params = state.get("tool_params", {})

    if not tool_name:
        return {
            "messages": [AIMessage(content="无工具调用")],
            "session_status": "idle",
            "tool_results": [],
            "slots": {},
        }

    logger.info(f"Tool call: {tool_name}, params: {tool_params}")

    new_slots = {}
    if "order_id" in tool_params:
        new_slots["order_id"] = tool_params["order_id"]
    if "phone" in tool_params:
        new_slots["phone"] = tool_params["phone"]

    tool_func = TOOL_MAP.get(tool_name)
    if tool_func:
        try:
            tool_result = tool_func.invoke(tool_params)
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
            "slots": {},
        }


# ==================== 构建 Graph ====================


def create_agent_graph():
    from langgraph.graph import StateGraph, END
    from langgraph.constants import Send

    graph = StateGraph(AgentState)

    graph.add_node("rag", rag_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("agent")

    def need_route(state: AgentState) -> str:
        tool_name = state.get("tool_name", "")
        if tool_name:
            return "tools"
        if state.get("need_rag"):
            return "rag"
        return "end"

    graph.add_conditional_edges(
        "agent",
        need_route,
        {
            "tools": "tools",
            "rag": "rag",
            "end": END,
        },
    )

    graph.add_edge("rag", END)
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
    need_rag = dispatch_result.need_rag
    need_tool = dispatch_result.need_tool
    need_clarify = dispatch_result.need_clarify
    tool_name = dispatch_result.tool_name
    tool_params = dispatch_result.tool_params

    logger.info(
        f"LLM dispatch: need_rag={need_rag}, need_tool={need_tool}, "
        f"tool_name={tool_name}, tool_params={tool_params}"
    )

    intent = "product" if need_rag else "order"
    session_status = "waiting" if need_clarify else "idle"

    slots = extract_all_slots(user_input)
    if tool_params:
        slots = {**slots, **tool_params}
    merged_slots = {**saved_slots, **slots}
    new_context_entity = update_context_entity(merged_slots, saved_context_entity)

    turn_count = prev_turn_count + 1

    messages = [HumanMessage(content=user_input)]

    initial_state = {
        "messages": messages,
        "session_id": session_id,
        "intent": intent,
        "rag_docs": [],
        "tool_results": [],
        "slots": merged_slots,
        "context_entity": new_context_entity,
        "session_status": session_status,
        "turn_count": turn_count,
        "tool_name": tool_name,
        "tool_params": tool_params,
    }

    if need_clarify:
        clarify_prompt = dispatch_result.clarify_prompt or "请问您具体想咨询什么？"
        yield clarify_prompt
        memory.add_user_message(user_input)
        memory.add_ai_message(clarify_prompt)
        memory.set_intent(intent)
        memory.add_slots(merged_slots)
        memory.add_context_entity(new_context_entity)
        memory.set_session_status("waiting")
        memory.increment_turn()
        return
        memory.increment_turn()
        return

    if not need_rag and not need_tool:
        chat_greetings = {"你好", "您好", "hi", "hello", "在吗", "在么", "有人吗", "哈喽"}
        chat_thanks = {"谢谢", "感谢", "谢了", "多谢"}
        chat_goodbyes = {"再见", "拜拜", "bye", "88", "回头聊"}
        user_lower = user_input.strip().lower()
        if any(word in user_lower for word in chat_greetings):
            chat_response = "您好！我是智能客服，有什么可以帮您的吗？😊"
        elif any(word in user_lower for word in chat_thanks):
            chat_response = "不客气！很高兴能帮到您~"
        elif any(word in user_lower for word in chat_goodbyes):
            chat_response = "再见！感谢您的咨询，祝您生活愉快！👋"
        else:
            chat_response = "您好！我可以帮您查询产品、售后、订单、物流等问题，请问需要什么帮助？"
        yield chat_response
        memory.add_user_message(user_input)
        memory.add_ai_message(chat_response)
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

    if need_rag:
        try:
            rag = get_rag()
            docs = rag.multi_search(user_input, k=2, vector_k=5, bm25_k=5)
            final_rag_docs = [d.page_content for d in docs]
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            try:
                docs = rag.similarity_search(user_input, k=2)
                final_rag_docs = [d.page_content for d in docs]
            except Exception as e2:
                final_rag_docs = []

    if need_tool and tool_name:
        try:
            tools_result = tools_node(initial_state)
            if "tool_results" in tools_result:
                final_tool_results = tools_result["tool_results"]
            if "slots" in tools_result and tools_result["slots"]:
                final_slots = {**final_slots, **tools_result["slots"]}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            final_tool_results = [{"name": tool_name, "result": "服务暂时不可用"}]

    final_session_status = "idle"

    llm_messages = [
        SystemMessage(content="你是机械革命官方客服，仅使用提供的资料回答，不编造。")
    ]

    if history:
        llm_messages.extend(history)

    rag_context = "\n".join(final_rag_docs) if need_rag else ""
    if rag_context:
        llm_messages.append(HumanMessage(content=f"参考资料：\n{rag_context}"))

    if final_tool_results:
        tool_context = "\n".join([
            f"工具结果 - {t['name']}: {t['result']}"
            for t in final_tool_results
        ])
        llm_messages.append(HumanMessage(content=f"工具查询结果：\n{tool_context}"))

    llm_messages.append(HumanMessage(content=user_input))

    final_text = ""
    for msg in llm_with_tools.stream(llm_messages):
        if msg.content:
            if enable_stream:
                for char in msg.content:
                    yield char
            final_text += msg.content

    final_response_text = final_text

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
