"""
意图识别与调度模块

基于 LLM 的动态调度决策
支持：RAG / Tool / 混合 / 追问 / 转人工
"""

import json
import re
import logging
from typing import Optional, List, Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config.logger import get_logger
from src.models.types import AgentState, DispatchResult

logger = get_logger(__name__)

DISPATCHER_PROMPT = """你是一个智能客服调度专家。根据用户输入和上下文，做出调度决策。

决策维度（全部必填）：
- need_rag: 是否需要 RAG 检索产品/售前/售后信息
- need_tool: 是否需要调用工具查询订单/物流/用户
- need_clarify: 是否需要追问澄清用户意图
- need_transfer: 是否直接转人工
- tool_call: 如果 need_tool=true，必须输出具体工具调用（name + args）

输出格式（JSON，必须严格按格式）：
{{
    "need_rag": true/false,
    "need_tool": true/false,
    "need_clarify": false,
    "need_transfer": false,
    "tool_call": {{
        "name": "query_order",
        "args": {{"order_id": "xxxx", "phone": "xxxx"}}
    }},
    "clarify_prompt": "",
    "reason": "决策理由"
}}

决策规则：
1. 产品/售前/售后咨询 → need_rag=true
2. 订单/物流/用户查询 → need_tool=true（必须同时输出 tool_call）
3. 两者都有 → need_rag=true, need_tool=true（tool_call 只给 need_tool=true 的场景）
4. 意图不明/信息不足 → need_clarify=true
5. 投诉/转人工/复杂问题 → need_transfer=true
6. 轮次>=8 → 优先转人工
7. tool_call 只在 need_tool=true 时必填，其他情况为 null
8. 返回纯 JSON，不要其他内容"""


def _build_context(state: AgentState) -> str:
    """构建上下文信息"""
    messages = state["messages"]
    history_parts = []

    for msg in messages[:-1]:
        if hasattr(msg, "content") and msg.content:
            role = "用户" if msg.type == "human" else "客服"
            history_parts.append(f"{role}: {msg.content}")

    history = "\n".join(history_parts[-6:]) if history_parts else "无历史对话"

    slots = state.get("slots", {})
    task_state = state.get("task_state", "pending")
    turn_count = state.get("turn_count", 0)

    slots_str = json.dumps(slots, ensure_ascii=False) if slots else "无"

    context = f"""历史对话：{history}
已提取槽位：{slots_str}
任务状态：{task_state}
对话轮次：{turn_count}"""

    return context


def _parse_dispatch_result(response_content: str) -> DispatchResult:
    """解析 LLM 返回的调度结果"""
    try:
        content = response_content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)

        tool_call_data = data.get("tool_call")
        if tool_call_data:
            tool_call = {
                "name": tool_call_data.get("name", ""),
                "args": tool_call_data.get("args", {})
            }
        else:
            tool_call = None

        return DispatchResult(
            need_rag=bool(data.get("need_rag", False)),
            need_tool=bool(data.get("need_tool", False)),
            need_clarify=bool(data.get("need_clarify", False)),
            need_transfer=bool(data.get("need_transfer", False)),
            clarify_prompt=data.get("clarify_prompt", ""),
            reason=data.get("reason", ""),
            tool_call=tool_call,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(
            f"Failed to parse dispatch result: {e}, content: {response_content}"
        )
        raise ValueError(f"LLM 调度结果解析失败: {e}")


def llm_dispatch(state: AgentState) -> DispatchResult:
    """
    LLM 动态调度决策

    Args:
        state: AgentState

    Returns:
        DispatchResult: 调度决策结果

    Raises:
        ValueError: LLM 调度失败时抛出异常
    """
    messages = state["messages"]
    if not messages:
        raise ValueError("LLM 调度失败: messages 为空")

    user_input = messages[-1].content
    if not user_input or not user_input.strip():
        raise ValueError("LLM 调度失败: 用户输入为空")

    context = _build_context(state)
    prompt = DISPATCHER_PROMPT.format(context=context, user_input=user_input)

    from src.services.llm import get_llm

    llm = get_llm()

    try:
        response = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=user_input)]
        )
    except Exception as e:
        logger.error(f"LLM dispatch invoke failed: {e}")
        raise ValueError(f"LLM 调度失败: {e}")

    if not hasattr(response, "content") or not response.content:
        raise ValueError("LLM 调度失败: 返回内容为空")

    dispatch_result = _parse_dispatch_result(response.content)

    logger.info(
        f"LLM dispatch: need_rag={dispatch_result['need_rag']}, "
        f"need_tool={dispatch_result['need_tool']}, "
        f"need_clarify={dispatch_result['need_clarify']}, "
        f"need_transfer={dispatch_result['need_transfer']}, "
        f"reason={dispatch_result['reason']}"
    )

    return dispatch_result


def get_routes_from_dispatch(dispatch_result: DispatchResult) -> List[str]:
    """从调度结果获取路由列表"""
    routes = []

    if dispatch_result.get("need_transfer"):
        return ["transfer"]

    if dispatch_result.get("need_clarify"):
        return ["clarify"]

    if dispatch_result.get("need_rag"):
        routes.append("rag")

    if dispatch_result.get("need_tool"):
        routes.append("agent")

    if not routes:
        routes.append("chat")

    return routes


def get_rag_filter_from_intent(intent: str) -> Optional[dict]:
    """根据意图获取 RAG filter"""
    filter_map = {
        "product": {"type": "product"},
        "pre_sales": {"type": "pre_sales"},
        "after_sales": {"type": "after_sales"},
    }
    return filter_map.get(intent)


PRODUCT_KEYWORDS = [
    "耀世",
    "蛟龙",
    "极光",
    "无界",
    "旷世",
    "翼龙",
    "深海",
    "泰坦",
    "小白",
]

FAULT_KEYWORDS = {
    "蓝屏": "蓝屏",
    "死机": "死机",
    "黑屏": "黑屏",
    "无法开机": "无法开机",
    "充电": "充电问题",
    "电池": "电池问题",
    "发热": "发热问题",
    "风扇": "风扇问题",
    "花屏": "花屏",
    "闪退": "闪退",
    "卡顿": "卡顿",
    "重装": "重装系统",
    "驱动": "驱动问题",
}


def extract_order_id(text: str) -> Optional[str]:
    """从文本中提取订单号"""
    text = text.lower()
    patterns = [
        r"订单号[：:\s]*(\d{10,})",
        r"(\d{10,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def extract_phone(text: str) -> Optional[str]:
    """从文本中提取手机号"""
    match = re.search(r"1[3-9]\d{9}", text)
    if match:
        return match.group()
    match = re.search(r"(手机|电话)[：:]*\s*(1[3-9]\d{9})", text)
    if match:
        return match.group(2)
    return None


def extract_product(text: str) -> Optional[str]:
    """从文本中提取产品型号"""
    text = text.lower()
    for product in PRODUCT_KEYWORDS:
        if product.lower() in text:
            return product
    return None


def extract_fault_type(text: str) -> Optional[str]:
    """从文本中提取故障类型"""
    text = text.lower()
    for keyword, fault_type in FAULT_KEYWORDS.items():
        if keyword in text:
            return fault_type
    return None


def extract_all_slots(text: str) -> dict:
    """一次性提取所有槽位"""
    slots = {}
    order_id = extract_order_id(text)
    phone = extract_phone(text)
    product = extract_product(text)
    fault_type = extract_fault_type(text)

    if order_id:
        slots["order_id"] = order_id
    if phone:
        slots["phone"] = phone
    if product:
        slots["product"] = product
    if fault_type:
        slots["fault_type"] = fault_type

    return slots


COREFERENCE_PATTERNS = [
    "那款",
    "这款",
    "它",
    "刚才那个",
    "刚才那个订单",
    "刚才那个产品",
    "上一个",
    "上一个订单",
    "上一个产品",
]


def resolve_coreference(text: str, context_entity: dict) -> str:
    """
    指代消解
    将"那款"、"它"等指代词消解为具体实体

    Args:
        text: 用户输入文本
        context_entity: 上下文实体字典

    Returns:
        消解后的文本
    """
    if not context_entity:
        return text

    text_lower = text.lower()

    for pattern in COREFERENCE_PATTERNS:
        if pattern in text_lower:
            if "product" in context_entity or "order" in context_entity:
                result = text
                if context_entity.get("last_product"):
                    result = result.replace(pattern, context_entity["last_product"])
                if context_entity.get("last_order"):
                    result = result.replace(pattern, context_entity["last_order"])
                if context_entity.get("last_phone"):
                    result = result.replace(pattern, context_entity["last_phone"])
                return result

    return text


def update_context_entity(slots: dict, context_entity: dict) -> dict:
    """
    更新上下文实体
    基于当前轮次提取的槽位更新上下文实体

    Args:
        slots: 当前提取的槽位
        context_entity: 当前上下文实体

    Returns:
        更新后的上下文实体
    """
    import time

    new_entity = dict(context_entity)

    if slots.get("product"):
        new_entity["last_product"] = slots["product"]
        new_entity["last_product_time"] = int(time.time())

    if slots.get("order_id"):
        new_entity["last_order"] = slots["order_id"]
        new_entity["last_order_time"] = int(time.time())

    if slots.get("phone"):
        new_entity["last_phone"] = slots["phone"]
        new_entity["last_phone_time"] = int(time.time())

    return new_entity


INTERRUPT_PATTERNS = [
    "算了",
    "先不提了",
    "换个话题",
    "暂停",
    "先这样",
    "不要了",
    "不提这个了",
    "换一个",
]


def is_interrupt(text: str) -> bool:
    """检测用户是否打断对话"""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in INTERRUPT_PATTERNS)
