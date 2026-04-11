"""
意图识别模块

基于规则的意图识别 + BERT 语义分类器兜底
"""

import re
import os
from enum import Enum
from typing import Optional


class Intent(Enum):
    """意图枚举"""

    ORDER_QUERY = "query_order"  # 订单查询
    LOGISTICS_QUERY = "query_logistics"  # 物流查询
    TRANSFER_HUMAN = "transfer"  # 转人工
    GENERAL_CHAT = "chat"  # 一般聊天
    UNKNOWN = "unknown"  # 未知


# 意图关键词映射
INTENT_PATTERNS = {
    Intent.ORDER_QUERY: [
        r"订单[号码]?[是]?\d+",
        r"查.{0,3}订单",
        r"订单状态",
        r"订单详情",
        r"订单信息",
        r"我的订单",
        r"订单\d{10,}",
        r"(?:order|订单).*?(?:查询|号|状态)",
    ],
    Intent.LOGISTICS_QUERY: [
        r"物流[信息]?",
        r"快递[信息]?",
        r"运输[信息]?",
        r"发货",
        r"到哪[里了]?",
        r"物流状态",
        r"快递单号",
        r"查.{0,3}物流",
        r"(?:logistics|快递).*?(?:查询|单号)",
    ],
    Intent.TRANSFER_HUMAN: [
        r"转人工",
        r"转接人工",
        r"人工客服",
        r"客服电话",
        r"我要投诉",
        r"找真人",
        r"human.*service",
        r"transfer.*human",
    ],
}

# 是否启用 BERT 兜底
USE_BERT_FALLBACK = os.getenv("USE_BERT_FALLBACK", "true").lower() == "true"


def recognize_intent(text: str) -> Intent:
    """
    识别用户意图（规则 + BERT 兜底）

    Args:
        text: 用户输入文本

    Returns:
        识别的意图
    """
    text_lower = text.lower()

    # 1. 规则匹配
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return intent

    # 2. BERT 兜底
    if USE_BERT_FALLBACK:
        try:
            from src.services.bert_classifier import predict_intent

            bert_intent = predict_intent(text)
            # 映射 BERT 结果到 Intent 枚举
            intent_map = {
                "query_order": Intent.ORDER_QUERY,
                "query_logistics": Intent.LOGISTICS_QUERY,
                "transfer": Intent.TRANSFER_HUMAN,
                "chat": Intent.GENERAL_CHAT,
            }
            return intent_map.get(bert_intent, Intent.GENERAL_CHAT)
        except Exception as e:
            # 如果 BERT 失败，返回默认意图
            from src.config.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(f"BERT fallback failed: {e}")

    return Intent.GENERAL_CHAT


def extract_order_id(text: str) -> Optional[str]:
    """
    从文本中提取订单号

    Args:
        text: 用户输入文本

    Returns:
        订单号，如果未找到返回 None
    """
    # 优先匹配10位以上数字
    match = re.search(r"\d{10,}", text)
    if match:
        return match.group()

    # 尝试匹配其他格式
    match = re.search(r"(?:订单号?|order)[：:]\s*(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def should_use_tools(text: str) -> bool:
    """
    判断是否需要使用工具

    Args:
        text: 用户输入文本

    Returns:
        是否需要使用工具
    """
    intent = recognize_intent(text)
    return intent in [Intent.ORDER_QUERY, Intent.LOGISTICS_QUERY, Intent.TRANSFER_HUMAN]


def get_intent_description(text: str) -> str:
    """
    获取意图描述（用于调试）

    Args:
        text: 用户输入文本

    Returns:
        意图描述
    """
    intent = recognize_intent(text)
    return intent.value
