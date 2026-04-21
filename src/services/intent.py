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

DISPATCHER_PROMPT = """你是机械革命笔记本客服调度专家，仅负责业务分流决策，严格输出指定JSON，不要多余文字。

全局硬性约束：
本对话仅限机械革命笔记本电脑业务。
用户提及的所有产品词汇、简称、模糊称呼，一律默认为机械革命旗下电脑型号，**禁止联想其他无关领域事物**。
用户未主动提及订单、物流、购买记录、售后单号、手机号时，不属于个人查询场景，**绝对不询问订单号，不触发订单相关逻辑**。


分流决策规则：
1. 用户明确提及订单、物流、购买记录、售后信息 → need_tool=true
2. 其余所有产品相关咨询、模糊机型称呼、参数疑问 → need_rag=true
3. 仅对话轮次超限、用户强烈投诉复杂问题时、用户发送敏感词（敏感词包括但不限于政治、暴力、违法等内容） → need_transfer=true
4. 产品类模糊意图不开启追问(need_clarify=false)，直接放行知识库检索，由RAG匹配具体机型。
5. 用户发送问候词不用检索知识库直接回复。
6. 用户单独输入纯数字，默认为订单号，按订单业务处理。

仅输出一行标准JSON，包含全部字段，无多余文字、无换行、无代码块。
{"need_rag":false,"need_tool":false,"need_clarify":false,"need_transfer":false,"tool_call":null,"clarify_prompt":"","reason":"分流原因"}"""



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
    try:
        content = response_content.strip()

        # 清理 markdown
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        # 【最强清洗】不管多少换行、空格、缩进，全部抓到 { ... }
        import re
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            content = json_match.group()

        # 【关键】Pydantic 直接解析，自带超强容错
        return DispatchResult.model_validate_json(content)

    except Exception as e:
        logger.warning(f"解析失败，使用兜底: {str(e)}")
        # 永远不崩溃
        return DispatchResult(
            need_rag=False,
            need_tool=False,
            need_clarify=False,
            need_transfer=False,
            tool_call=None,
            clarify_prompt="",
            reason="解析失败兜底"
        )


def llm_dispatch(state: AgentState) -> DispatchResult:
    try:
        messages = state["messages"]
        if not messages:
            return DispatchResult(need_rag=False, need_tool=False, need_clarify=False, need_transfer=False, tool_call=None, clarify_prompt="", reason="无消息")

        user_input = messages[-1].content
        if not user_input or not user_input.strip():
            return DispatchResult(need_rag=False, need_tool=False, need_clarify=False, need_transfer=False, tool_call=None, clarify_prompt="", reason="无输入")

        prompt = DISPATCHER_PROMPT

        from src.services.llm import get_llm
        llm = get_llm()


        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_input)
        ])

        if not hasattr(response, "content") or not response.content or response.content.strip() == "":
            logger.warning("模型返回空，使用兜底决策")
            return DispatchResult(
                need_rag=False, need_tool=False,
                need_clarify=False, need_transfer=False,
                tool_call=None, clarify_prompt="", reason="空返回兜底"
            )

        return _parse_dispatch_result(response.content)

    except Exception as e:
        logger.error(f"调度异常: {e}")
        return DispatchResult(
            need_rag=False, need_tool=False,
            need_clarify=False, need_transfer=False,
            tool_call=None, clarify_prompt="", reason="异常兜底"
        )

def get_routes_from_dispatch(dispatch_result: DispatchResult) -> List[str]:
    """从调度结果获取路由列表"""
    routes = []

    if dispatch_result.need_transfer:
        return ["transfer"]

    if dispatch_result.need_clarify:
        return ["clarify"]

    if dispatch_result.need_rag:
        routes.append("rag")

    if dispatch_result.need_tool:
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
