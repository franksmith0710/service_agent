"""
Agent 工具模块

提供订单查询、物流查询、转人工等工具
"""

import logging
import re
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def query_order(order_id: str, user_id: Optional[str] = None) -> str:
    """
    查询订单信息

    Args:
        order_id: 订单号
        user_id: 用户ID（可选）

    Returns:
        订单信息字符串
    """
    logger.info(f"Querying order: {order_id}")

    order_info = {
        "order_id": order_id,
        "status": "shipped",
        "items": ["Product A x1", "Product B x2"],
        "total_amount": 299.00,
        "created_at": "2024-01-15 10:30:00",
    }
    return str(order_info)


@tool
def query_logistics(order_id: str) -> str:
    """
    查询物流信息

    Args:
        order_id: 订单号

    Returns:
        物流信息字符串
    """
    logger.info(f"Querying logistics: {order_id}")

    logistics_info = {
        "order_id": order_id,
        "carrier": "SF Express",
        "tracking_number": "SF1234567890",
        "status": "in_transit",
        "trace": [
            {
                "time": "2024-01-16 14:00",
                "location": "Shanghai",
                "status": "dispatched",
            },
            {
                "time": "2024-01-15 20:00",
                "location": "Beijing warehouse",
                "status": "picked up",
            },
        ],
    }
    return str(logistics_info)


@tool
def transfer_to_human(reason: str, conversation_summary: Optional[str] = None) -> str:
    """
    转接人工客服

    Args:
        reason: 转接原因
        conversation_summary: 对话摘要（可选）

    Returns:
        转接结果字符串
    """
    logger.info(f"Transfer to human: {reason}")

    result = {
        "status": "success",
        "message": "Transferred to human agent. Please wait.",
        "ticket_id": "TK20240115001",
    }
    return str(result)


def get_all_tools() -> list:
    """获取所有工具"""
    return [query_order, query_logistics, transfer_to_human]


def extract_order_id(text: str) -> Optional[str]:
    """
    从文本中提取订单号

    Args:
        text: 输入文本

    Returns:
        订单号，如果未找到返回 None
    """
    match = re.search(r"\d{10,}", text)
    return match.group() if match else None
