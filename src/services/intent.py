"""
意图识别与调度模块

基于 LLM 的动态调度决策
支持：RAG / Tool / 混合 / 追问 / 转人工
"""

import re
import logging
from typing import Optional, List, Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config.logger import get_logger
from src.models.types import AgentState, DispatchResult

logger = get_logger(__name__)

DISPATCHER_PROMPT = """你是机械革命笔记本客服调度专家，仅做业务分流决策。

全局硬性约束：
本对话仅限机械革命笔记本电脑业务。
用户提及的所有产品词汇、简称、模糊称呼，一律默认为机械革命旗下电脑型号，禁止联想其他无关领域事物。
分流决策规则：
1. 用户明确提及订单、物流、购买记录、售后信息 → need_tool=true，tool_name="query_order"或"query_logistics"
2. 用户明确要求转人工 → need_tool=true，tool_name="transfer_to_human"
3. 其余所有产品相关咨询、模糊机型称呼、参数疑问 → need_rag=true
4. 用户询问产品但信息不全 → need_clarify=true，并给出clarify_prompt
5. 用户问候无需检索，直接兜底回复
6. 用户输入纯数字默认为订单号，触发tool_name="query_order"

参数提取规则：
- query_order参数：order_id 或 phone，从用户输入中提取
- query_logistics参数：order_id 或 phone
- transfer_to_human参数：reason（转接原因）

要求：
1. 输出标准合法JSON，tool_params字段统一为{}，禁止使用null
2. 只输出一行纯JSON，无多余文字、无换行、无解释
3. 必须完整输出，绝对不允许空输出

输出模板（严格照搬）：
{"need_rag":false,"need_tool":false,"need_clarify":false,"tool_name":"","tool_params":{},"clarify_prompt":""}
"""


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
        return DispatchResult(
            need_rag=False,
            need_tool=False,
            need_clarify=False,
            tool_name="",
            tool_params={},
            clarify_prompt="",
        )


def llm_dispatch(state: AgentState) -> DispatchResult:
    try:
        messages = state["messages"]
        if not messages:
            return DispatchResult(need_rag=False, need_tool=False, need_clarify=False, tool_name="", tool_params={}, clarify_prompt="")

        user_input = messages[-1].content
        if not user_input or not user_input.strip():
            return DispatchResult(need_rag=False, need_tool=False, need_clarify=False, tool_name="", tool_params={}, clarify_prompt="")

        history = messages[:-1]

        prompt = DISPATCHER_PROMPT

        from src.services.llm import get_llm
        llm = get_llm()

        msgs = [SystemMessage(content=prompt)]
        msgs.extend(history)
        msgs.append(HumanMessage(content=user_input))

        response = llm.invoke(msgs)

        if not hasattr(response, "content") or not response.content or response.content.strip() == "":
            logger.warning("模型返回空，使用兜底决策")
            return DispatchResult(
                need_rag=False, need_tool=False,
                need_clarify=False, tool_name="",
                tool_params={}, clarify_prompt="",
            )

        return _parse_dispatch_result(response.content)

    except Exception as e:
        logger.error(f"调度异常: {e}")
        return DispatchResult(
            need_rag=False, need_tool=False,
            need_clarify=False, tool_name="",
            tool_params={}, clarify_prompt="",
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
