"""
意图识别模块

基于关键词精确匹配的意图识别
优先级：CHAT > TRANSFER > PRODUCT > ORDER > LOGISTICS > USER > UNKNOWN
"""

import re
from enum import Enum
from typing import Optional

from src.config.logger import get_logger

logger = get_logger(__name__)


class Intent(Enum):
    """意图枚举"""

    CHAT = "chat"  # 问候闲聊
    TRANSFER_HUMAN = "transfer"  # 转人工
    PRODUCT_QUERY = "product"  # 产品咨询
    PRE_SALES_QUERY = "pre_sales"  # 售前咨询
    AFTER_SALES_QUERY = "after_sales"  # 售后咨询
    ORDER_QUERY = "order"  # 订单查询
    LOGISTICS_QUERY = "logistics"  # 物流查询
    USER_QUERY = "user"  # 用户信息查询
    UNKNOWN = "unknown"  # 未知


# ============ 关键词匹配（按优先级排序） ============

INTENT_KEYWORDS = {
    # 最高优先级：问候闲聊
    Intent.CHAT: [
        "你好",
        "您好",
        "嗨",
        "hi",
        "hello",
        "在吗",
        "谢谢",
        "感谢",
        "再见",
        "bye",
        "拜拜",
        "早上好",
        "晚上好",
        "晚安",
    ],
    # 高优先级：转人工
    Intent.TRANSFER_HUMAN: [
        "转人工",
        "转接人工",
        "人工客服",
        "找真人",
        "投诉",
        "差评",
        "我要投诉",
        "客服电话",
        "找客服",
        "找领导",
        "找老板",
    ],
    # 中高优先级：产品咨询（先于订单，防止误匹配）
    Intent.PRODUCT_QUERY: [
        "商品",
        "产品",
        "介绍",
        "推荐",
        "配置",
        "怎么样",
        "好不好",
        "值不值",
        "性价比",
        "蛟龙",
        "极光",
        "耀世",
        "无界",
        "旷世",
        "翼龙",
        "深海",
        "电脑",
        "笔记本",
        "游戏本",
        "轻薄本",
        "cpu",
        "显卡",
        "内存",
        "硬盘",
        "屏幕",
        "价格",
        "多少钱",
        "报价",
        "优惠",
    ],
    # 售前咨询（保修政策、选购、驱动等）
    Intent.PRE_SALES_QUERY: [
        "选购",
        "推荐",
        "适合",
        "哪款",
        "系列",
        "性价比",
        "蛟龙",
        "极光",
        "耀世",
        "无界",
        "旷世",
        "翼龙",
        "深海",
    ],
    # 售后咨询
    Intent.AFTER_SALES_QUERY: [
        "保修",
        "售后",
        "维修",
        "驱动",
        "重装系统",
        "蓝屏",
        "死机",
        "黑屏",
        "开机",
        "充电",
        "电池",
        "发热",
        "风扇",
        "温度",
        "故障",
    ],
    # 中优先级：订单查询
    Intent.ORDER_QUERY: [
        "订单号",
        "订单状态",
        "订单详情",
        "我的订单",
        "查订单",
        "看订单",
        "订单查询",
    ],
    # 中优先级：物流查询
    Intent.LOGISTICS_QUERY: [
        "物流",
        "快递",
        "发货",
        "到哪了",
        "物流信息",
        "快递单号",
        "物流单号",
        "发货了吗",
    ],
    # 低优先级：用户信息
    Intent.USER_QUERY: [
        "会员",
        "积分",
        "用户信息",
        "我的资料",
        "账号",
        "个人信息",
        "等级",
    ],
}


def _normalize_text(text: str) -> str:
    """标准化文本"""
    text = text.lower().strip()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("？", "?").replace("！", "!")
    text = re.sub(r"\s+", " ", text)
    return text


def _contains_keyword(text: str, keywords: list[str]) -> bool:
    """检查是否包含关键词"""
    for kw in keywords:
        if kw.lower() in text:
            return True
    return False


def recognize_intent(text: str) -> Intent:
    """
    识别用户意图（关键词精确匹配）

    优先级：CHAT > TRANSFER > PRODUCT > ORDER > LOGISTICS > USER > UNKNOWN

    Args:
        text: 用户输入文本

    Returns:
        识别的意图
    """
    text = _normalize_text(text)

    # 1. 问候闲聊
    if _contains_keyword(text, INTENT_KEYWORDS[Intent.CHAT]):
        return Intent.CHAT

    # 2. 转人工
    if _contains_keyword(text, INTENT_KEYWORDS[Intent.TRANSFER_HUMAN]):
        return Intent.TRANSFER_HUMAN

    # 3. 产品咨询（先于订单）
    if _contains_keyword(text, INTENT_KEYWORDS[Intent.PRODUCT_QUERY]):
        return Intent.PRODUCT_QUERY

    # 4. 售后咨询
    if _contains_keyword(text, INTENT_KEYWORDS[Intent.AFTER_SALES_QUERY]):
        return Intent.AFTER_SALES_QUERY

    # 5. 售前咨询
    if _contains_keyword(text, INTENT_KEYWORDS[Intent.PRE_SALES_QUERY]):
        return Intent.PRE_SALES_QUERY

    # 6. 订单查询
    if _contains_keyword(text, INTENT_KEYWORDS[Intent.ORDER_QUERY]):
        return Intent.ORDER_QUERY

    # 5. 物流查询
    if _contains_keyword(text, INTENT_KEYWORDS[Intent.LOGISTICS_QUERY]):
        return Intent.LOGISTICS_QUERY

    # 6. 用户信息
    if _contains_keyword(text, INTENT_KEYWORDS[Intent.USER_QUERY]):
        return Intent.USER_QUERY

    return Intent.UNKNOWN


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


def should_use_tools(text: str) -> bool:
    """判断是否需要使用工具"""
    intent = recognize_intent(text)
    return intent in [Intent.ORDER_QUERY, Intent.LOGISTICS_QUERY, Intent.USER_QUERY]


def get_intent_description(text: str) -> str:
    """获取意图描述"""
    intent = recognize_intent(text)
    return intent.value


def get_rag_filter(intent: Intent) -> Optional[dict]:
    """根据意图获取 RAG 检索过滤条件

    Args:
        intent: 意图枚举

    Returns:
        Chroma filter dict or None
    """
    filter_map = {
        Intent.PRODUCT_QUERY: {"type": "product"},
        Intent.PRE_SALES_QUERY: {"type": "pre_sales"},
        Intent.AFTER_SALES_QUERY: {"type": "after_sales"},
    }
    return filter_map.get(intent)
