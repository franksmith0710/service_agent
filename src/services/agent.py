"""
Agent 核心服务模块

采用两级 LLM 架构：
- 决策 LLM：意图识别、路由决策（run_agent 入口调用）
- 执行层：Graph 流程处理 RAG 检索 + 工具执行（零 LLM 调用）
- 生成 LLM：整合所有素材生成最终回答（run_agent 末尾调用）
"""

import os
import logging
from typing import Generator

from langchain_core.messages import (
    HumanMessage,
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
from src.services.prompts import (
    GREETINGS,
    THANKS,
    GOODBYES,
    CHAT_GREETING,
    CHAT_THANKS,
    CHAT_GOODBYE,
    CHAT_DEFAULT,
    CLARIFY_ORDER,
    CLARIFY_DEFAULT,
    GENERATION_SYSTEM_PROMPT,
)
from src.services.rag import get_rag

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

from src.services.llm import get_llm_for_generation


# ==================== 节点函数 ====================

def safe_node(func):
    """全链路异常兜底装饰器"""
    def wrapper(state):
        try:
            return func(state)
        except Exception as e:
            logger.error(f"节点执行异常: {func.__name__}, {e}")
            return {}
    return wrapper


@safe_node
def rag_node(state: AgentState) -> AgentState:
    """RAG 检索节点 - 多路召回"""
    messages = state["messages"]
    user_query = messages[-1].content

    logger.info(f"RAG multi-search: {user_query[:30]}...")

    try:
        rag = get_rag()
        docs = rag.multi_search(user_query, k=2, vector_k=3, bm25_k=3)
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
    return {"rag_docs": rag_docs}


@safe_node
def check_slots_node(state: AgentState) -> AgentState:
    """
    槽位校验
    订单/物流必须有 order_id 或 phone，没有 → 强制追问
    """
    if state.get("need_clarify"):
        clarify_prompt = state.get("clarify_prompt", CLARIFY_ORDER)
        return {
            "need_clarify": True,
            "clarify_prompt": clarify_prompt,
            "step": "clarify",
            "next_step": "end"
        }

    intent = state.get("intent", "")
    slots = state.get("slots", {})

    if intent in ["order", "logistics"]:
        if not slots.get("order_id") and not slots.get("phone"):
            return {
                "need_clarify": True,
                "clarify_prompt": CLARIFY_ORDER,
                "step": "clarify",
                "next_step": "end"
            }

    return {
        "need_clarify": False,
        "step": "tools",
        "next_step": "tools"
    }


@safe_node
def clarify_node(state: AgentState) -> AgentState:
    """
    追问回复节点（仅输出提示，不执行工具）
    从 state 中读取 clarify_prompt 并存入返回 state
    """
    clarify_prompt = state.get("clarify_prompt", CLARIFY_DEFAULT)
    return {
        "clarify_prompt": clarify_prompt,
        "step": "end",
        "next_step": "end"
    }


@safe_node
def summary_node(state: AgentState) -> AgentState:
    tool_results = state.get("tool_results", [])
    rag_docs = state.get("rag_docs", [])

    # 脱敏函数（强制截断，单段最多 500 字符）
    def mask_sensitive(text):
        import re
        text = str(text)
        text = re.sub(r"1[3-9]\d{9}", "1**********", text)
        text = re.sub(r"\d{10,}", "**********", text)
        return text[:500]

    # 1. 先脱敏
    masked_tools = [
        {"name": t["name"], "result": mask_sensitive(t["result"])}
        for t in tool_results
    ]

    # 2. 同名工具**只保留最后一条**，彻底去重
    unique_tools = {}
    for item in masked_tools:
        unique_tools[item["name"]] = item
    masked_tools = list(unique_tools.values())

    # 3. RAG 文档去重（取前 2 条）
    rag_final = [doc for doc in rag_docs[:2]]

    # 直接返回全新结果，覆盖旧 state，不再拼接冗余
    return {
        "tool_results": masked_tools,
        "rag_docs": rag_final,
        "step": "end",
        "next_step": "end"
    }


MAX_TURNS = 10


@safe_node
def tools_node(state: AgentState) -> AgentState:
    """
    ReAct工具节点：单次执行一个工具
    配合循环节点实现多工具自动依次调用
    """
    tool_queue = state.get("tool_queue", [])
    exec_count = state.get("tool_exec_count", 0)
    max_limit = state.get("max_tool_limit", 4)
    old_results = state.get("tool_results", [])
    old_slots = state.get("slots", {})

    # 循环终止条件1：队列为空
    if not tool_queue:
        return {"tool_results": old_results, "slots": old_slots}

    # 循环终止条件2：超过最大调用次数
    if exec_count >= max_limit:
        logger.warning("已达到最大工具调用上限，终止工具循环")
        return {"tool_results": old_results, "slots": old_slots}

    # 取出当前第一个待执行工具
    current_tool = tool_queue[0]
    remain_queue = tool_queue[1:]
    tool_name = current_tool["name"]
    tool_args = current_tool["args"]

    tool_func = TOOL_MAP.get(tool_name)

    # 工具执行
    try:
        tool_result = tool_func.invoke(tool_args)
    except Exception as e:
        logger.error(f"Tool call failed: {tool_name}, {e}")
        tool_result = TOOL_ERROR_STRATEGY.get(tool_name, "服务暂时不可用")

    # 追加槽位、追加结果，不覆盖原有数据
    new_slots = dict(old_slots)
    if "order_id" in tool_args:
        new_slots["order_id"] = tool_args["order_id"]
    if "phone" in tool_args:
        new_slots["phone"] = tool_args["phone"]

    new_results = old_results + [{"name": tool_name, "result": tool_result}]

    return {
        "tool_results": new_results,
        "slots": new_slots,
        "tool_queue": remain_queue,
        "tool_exec_count": exec_count + 1,
    }


def tool_loop_check_node(state: AgentState) -> str:
    """
    循环路由判断节点
    工具跑完后判断：还有工具要执行吗？
    有 → 回到tools继续执行
    无 → 根据 need_rag 决定去 RAG 还是 summary
    """
    queue = state.get("tool_queue", [])
    exec_count = state.get("tool_exec_count", 0)
    max_limit = state.get("max_tool_limit", 4)
    need_rag = state.get("need_rag", False)

    # 还有工具 & 没超次数 → 继续循环执行工具
    if queue and exec_count < max_limit:
        return "tools"
    # 工具全部跑完 + 需要 RAG → 进入 RAG 检索
    elif need_rag:
        return "rag"
    # 工具全部跑完 + 不需要 RAG → 去汇总（脱敏处理）
    else:
        return "summary"


# ==================== 构建 Graph ====================


def create_agent_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)

    # 全企业节点
    graph.add_node("check_slots", check_slots_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("tools", tools_node)
    graph.add_node("rag", rag_node)
    graph.add_node("summary", summary_node)

    # 入口：槽位校验
    graph.set_entry_point("check_slots")

    # 路由：槽位校验完 → 追问 或 工具
    graph.add_conditional_edges(
        "check_slots",
        lambda s: "clarify" if s.get("need_clarify") else "tools",
        {
            "clarify": "clarify",
            "tools": "tools"
        }
    )
    # 追问直接结束
    graph.add_edge("clarify", END)

    # 工具循环：tools 执行完后，根据 tool_loop_check_node 判断下一步
    graph.add_conditional_edges(
        "tools",
        tool_loop_check_node,
        {
            "tools": "tools",    # 继续循环执行工具
            "rag": "rag",       # 工具跑完，进入 RAG
            "summary": "summary" # 工具跑完 + 不需要 RAG，去汇总
        }
    )

    # RAG → 汇总
    graph.add_edge("rag", "summary")
    graph.add_edge("summary", END)

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
        yield "对话已达最大轮次(10轮)，为保证服务质量已为您转接人工客服。\n人工客服工作时间: 周一至周五 9:00-18:00\n客服热线: 400-990-5898"
        memory.clear()
        memory.set_session_status("idle")
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
        "messages": history[-4:] + [HumanMessage(content=user_input)],
        "session_id": session_id,
        "slots": saved_slots,
        "context_entity": saved_context_entity,
        "turn_count": prev_turn_count,
    }

    dispatch_result = llm_dispatch(raw_state)
    need_rag = dispatch_result.need_rag
    need_tool = dispatch_result.need_tool
    need_clarify = dispatch_result.need_clarify
    tool_calls = dispatch_result.tool_calls
    clarify_prompt = dispatch_result.clarify_prompt

    logger.info(
        f"LLM dispatch: need_rag={need_rag}, need_tool={need_tool}, "
        f"tool_calls={tool_calls}"
    )

    if need_tool and tool_calls:
        tool_names = [tc.get("name") for tc in tool_calls]
        if "query_order" in tool_names or "query_logistics" in tool_names:
            intent = "order"
        elif "query_user_info" in tool_names:
            intent = "user"
        elif "transfer_to_human" in tool_names:
            intent = "transfer"
        else:
            intent = "tool"
    elif need_rag:
        intent = "product"
    else:
        intent = "chat"

    session_status = "waiting" if need_clarify else "idle"

    user_slots = extract_all_slots(user_input)

    tool_slots = {}
    for tc in tool_calls:
        args = tc.get("args", {})
        if args:
            tool_slots.update(args)
        if tc.get("name") == "transfer_to_human":
            args["user_id"] = saved_slots.get("user_id")
            args["phone"] = saved_slots.get("phone")
            args["session_id"] = session_id

    merged_slots = {**saved_slots, **tool_slots, **user_slots}
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
        "tool_calls": tool_calls,
        "need_rag": need_rag,
        "need_clarify": need_clarify,
        "clarify_prompt": clarify_prompt,

        # ========== 新增循环默认值 ==========
        "tool_queue": tool_calls,     # 调度返回的工具全部扔进待执行队列
        "tool_exec_count": 0,         # 初始已执行 0 个
        "max_tool_limit": 4,          # 最多允许调用 4 个工具

        # ========== 第三步新增 ==========
        "step": "init",
        "next_step": "check_slots",
    }

    if not need_rag and not need_tool and not need_clarify:
        user_lower = user_input.strip().lower()
        if any(word in user_lower for word in GREETINGS):
            chat_response = CHAT_GREETING
        elif any(word in user_lower for word in THANKS):
            chat_response = CHAT_THANKS
        elif any(word in user_lower for word in GOODBYES):
            chat_response = CHAT_GOODBYE
        else:
            chat_response = CHAT_DEFAULT
        yield chat_response
        memory.add_user_message(user_input)
        memory.add_ai_message(chat_response)
        memory.set_intent("chat")
        memory.set_session_status("idle")
        memory.add_slots(merged_slots)
        memory.add_context_entity(new_context_entity)
        memory.increment_turn()
        return

    final_clarify_prompt = None
    final_rag_docs = []
    final_tool_results = []
    final_session_status = "idle"
    current_intent = initial_state.get("intent")
    final_slots = initial_state.get("slots", {})
    final_context_entity = initial_state.get("context_entity", {})

    agent_graph = get_agent_graph()
    final_state = agent_graph.invoke(initial_state)

    # 从最终完整 state 读取结果
    final_clarify_prompt = final_state.get("clarify_prompt")
    final_rag_docs = final_state.get("rag_docs", [])
    final_tool_results = final_state.get("tool_results", [])
    final_slots = final_state.get("slots", {})

    # 检查是否需要追问
    if final_clarify_prompt:
        yield final_clarify_prompt
        memory.add_user_message(user_input)
        memory.add_ai_message(final_clarify_prompt)
        memory.set_intent(intent)
        memory.add_slots(merged_slots)
        memory.add_context_entity(new_context_entity)
        memory.set_session_status("waiting")
        memory.increment_turn()
        return

    final_session_status = "idle"

    llm_messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT)
    ]

    # 只保留最近2轮完整对话（4条消息），避免爆上下文
    recent_msgs = history[-4:]
    if recent_msgs:
        llm_messages.extend(recent_msgs)

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
    for msg in get_llm_for_generation().stream(llm_messages):
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
