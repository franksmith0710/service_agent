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

SIMPLIFIED_PROMPT = """你是机械革命笔记本客服调度专家，仅做业务分流决策。

分流决策规则：
1. 产品相关咨询、模糊机型称呼、参数疑问 → need_rag=true
2. 用户问候、闲聊 → need_rag=false, need_tool=false
3. 无法判断时 → need_rag=true

要求：
1. 输出标准合法JSON
2. 只输出一行纯JSON，无多余文字

输出模板：
{"need_rag":false,"need_tool":false,"need_clarify":false,"tool_calls":[],"clarify_prompt":""}
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
            tool_calls=[],
            clarify_prompt="",
        )


def llm_dispatch_by_mini_model(state: AgentState) -> DispatchResult:
    """小模型兜底：仅处理规则无法覆盖的模糊语句"""
    try:
        messages = state["messages"]
        if not messages:
            return DispatchResult(need_rag=False, need_tool=False, need_clarify=False, tool_calls=[], clarify_prompt="")

        user_input = messages[-1].content
        if not user_input or not user_input.strip():
            return DispatchResult(need_rag=False, need_tool=False, need_clarify=False, tool_calls=[], clarify_prompt="")

        history = messages[:-1]

        from src.services.llm import get_llm
        llm = get_llm()

        msgs = [SystemMessage(content=SIMPLIFIED_PROMPT)]
        msgs.extend(history)
        msgs.append(HumanMessage(content=user_input))

        response = llm.invoke(msgs)

        if not hasattr(response, "content") or not response.content or response.content.strip() == "":
            logger.warning("模型返回空，使用兜底决策")
            return DispatchResult(need_rag=True, need_tool=False, need_clarify=False, tool_calls=[], clarify_prompt="")

        return _parse_dispatch_result(response.content)

    except Exception as e:
        logger.error(f"小模型调度异常: {e}")
        return DispatchResult(need_rag=True, need_tool=False, need_clarify=False, tool_calls=[], clarify_prompt="")


def llm_dispatch(state: AgentState) -> DispatchResult:
    """
    企业级调度：规则优先 + 小模型兜底
    规则 100% 覆盖明确场景，不占用模型
    """
    try:
        messages = state["messages"]
        if not messages:
            return DispatchResult(need_rag=False, need_tool=False, need_clarify=False, tool_calls=[], clarify_prompt="")

        user_input = messages[-1].content
        if not user_input or not user_input.strip():
            return DispatchResult(need_rag=False, need_tool=False, need_clarify=False, tool_calls=[], clarify_prompt="")

        # 提取槽位（规则匹配）
        slots = extract_all_slots(user_input)
        user_input_lower = user_input.lower()

        # ====== 规则 1：明确有订单号或手机号 → 自动识别工具 ======
        order_id = extract_order_id(user_input)
        phone = extract_phone(user_input)

        if order_id or phone:
            tool_args = {}
            if order_id:
                tool_args["order_id"] = order_id
            if phone:
                tool_args["phone"] = phone

            # 企业规则：自动识别并添加多个工具
            tool_list = []

            # 有订单号 → 查订单 + 物流
            if order_id:
                tool_list.append({"name": "query_order", "args": tool_args})
                tool_list.append({"name": "query_logistics", "args": tool_args})

            # 有手机号 → 查用户信息
            if phone:
                tool_list.append({"name": "query_user_info", "args": tool_args})

            return DispatchResult(
                need_rag=False,
                need_tool=True,
                need_clarify=False,
                tool_calls=tool_list,  # 多工具自动进队列
                clarify_prompt=""
            )

        # ====== 规则 2：提到订单/物流但缺少信息 → 追问 ======
        if any(k in user_input_lower for k in ["订单", "物流", "查单", "快递"]):
            return DispatchResult(
                need_clarify=True,
                clarify_prompt="请提供您的订单号或手机号，以便为您查询",
                need_rag=False,
                need_tool=False,
                tool_calls=[]
            )

        # ====== 规则 3：提到产品相关 → RAG 检索 ======
        product_keywords = ["产品", "配置", "参数", "性能", "价格", "多少钱", "怎么样", "好不好"]
        if any(k in user_input_lower for k in product_keywords + [p.lower() for p in PRODUCT_KEYWORDS]):
            return DispatchResult(
                need_rag=True,
                need_tool=False,
                need_clarify=False,
                tool_calls=[],
                clarify_prompt=""
            )

        # ====== 规则 4：明确转人工 ======
        if any(k in user_input_lower for k in ["人工", "转人工", "投诉", "找人"]):
            return DispatchResult(
                need_tool=True,
                tool_calls=[{"name": "transfer_to_human", "args": {"reason": "用户主动要求转人工"}}],
                need_rag=False,
                need_clarify=False,
                clarify_prompt=""
            )

        # ====== 规则 5：简单问候 → 直接聊天 ======
        greetings = ["你好", "您好", "hi", "hello", "嗨"]
        thanks = ["谢谢", "感谢", "thx", "thanks"]
        if any(k in user_input_lower for k in greetings + thanks):
            return DispatchResult(
                need_rag=False,
                need_tool=False,
                need_clarify=False,
                tool_calls=[],
                clarify_prompt=""
            )

        # ====== 兜底：小模型处理模糊语句 ======
        logger.info("规则无法覆盖，调用小模型兜底")
        return llm_dispatch_by_mini_model(state)

    except Exception as e:
        logger.error(f"规则调度异常: {e}")
        return DispatchResult(
            need_rag=True,
            need_tool=False,
            need_clarify=False,
            tool_calls=[],
            clarify_prompt=""
        )


def get_routes_from_dispatch(dispatch_result: DispatchResult) -> List[str]:
    """从调度结果获取路由列表"""
    routes = []

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
