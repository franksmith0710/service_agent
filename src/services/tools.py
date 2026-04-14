"""
Agent 工具模块

提供订单查询、物流查询、转人工等工具
"""

import logging
import re
from typing import Optional
from langchain_core.tools import tool

from src.services.database import (
    get_order_by_id,
    get_order_by_phone,
    get_logistics_by_order,
    get_user_by_id,
    get_user_by_phone,
    search_orders,
)

logger = logging.getLogger(__name__)


def extract_order_id(text: str) -> Optional[str]:
    """从文本中提取订单号"""
    match = re.search(r"\d{10,}", text)
    return match.group() if match else None


def extract_phone(text: str) -> Optional[str]:
    """从文本中提取手机号"""
    match = re.search(r"1[3-9]\d{9}", text)
    return match.group() if match else None


@tool
def query_order(
    order_id: str = "", user_id: Optional[str] = None, phone: Optional[str] = None
) -> str:
    """
    查询订单信息

    Args:
        order_id: 订单号（优先）
        user_id: 用户ID（可选）
        phone: 手机号（可选）

    Returns:
        订单信息字符串
    """
    # 如果没有订单号，尝试从手机号查找
    if not order_id and phone:
        orders = get_order_by_phone(phone)
        if orders:
            order = orders[0]
            return _format_order(order)
        return "未查询到该手机号对应的订单"

    # 如果没有订单号，尝试搜索
    if not order_id:
        return "请提供订单号或手机号以便查询订单"

    # 查询订单
    order = get_order_by_id(order_id)
    if not order:
        # 尝试搜索
        orders = search_orders(order_id)
        if orders:
            return f"未找到订单号 {order_id}，为您找到以下相关订单：\n" + "\n".join(
                [
                    f"- 订单号: {o.order_id}, 状态: {o.status}, 金额: ¥{o.total_amount}"
                    for o in orders[:3]
                ]
            )
        return f"未找到订单号 {order_id} 的订单信息"

    return _format_order(order)


def _format_order(order) -> str:
    """格式化订单信息"""
    items_str = "\n".join(
        [
            f"  - {item['name']} x{item['quantity']} ¥{item['price']}"
            for item in order.items
        ]
    )
    return f"""订单号: {order.order_id}
状态: {order.status}
商品:
{items_str}
总金额: ¥{order.total_amount}
下单时间: {order.created_at}
支付方式: {order.pay_method}
收货地址: {order.shipping_address or "未填写"}"""


@tool
def query_logistics(order_id: str = "", phone: Optional[str] = None) -> str:
    """
    查询物流信息

    Args:
        order_id: 订单号（优先）
        phone: 手机号（可选）

    Returns:
        物流信息字符串
    """
    # 如果没有订单号，尝试从手机号查找最新订单
    if not order_id and phone:
        orders = get_order_by_phone(phone)
        if orders:
            # 找已发货的订单
            for o in orders:
                if o.status in ["已发货", "配送中"]:
                    order_id = o.order_id
                    break
            if not order_id:
                return "该手机号下没有已发货的订单"
        else:
            return "未查询到该手机号对应的订单"

    if not order_id:
        return "请提供订单号以便查询物流"

    # 查询物流
    logistics = get_logistics_by_order(order_id)
    if not logistics:
        # 查询订单信息
        order = get_order_by_id(order_id)
        if not order:
            return f"未找到订单号 {order_id} 的信息"
        if order.status in ["待发货", "已取消"]:
            return f"订单号 {order_id} 尚未发货，当前状态: {order.status}"
        return f"订单号 {order_id} 暂无物流信息"

    return _format_logistics(logistics)


def _format_logistics(logistics) -> str:
    """格式化物流信息"""
    trace_str = "\n".join(
        [f"  [{t['time']}] {t['location']} - {t['status']}" for t in logistics.trace]
    )
    return f"""订单号: {logistics.order_id}
承运商: {logistics.carrier}
运单号: {logistics.tracking_number}
当前状态: {logistics.status}

物流轨迹:
{trace_str}"""


@tool
def query_user_info(phone: str) -> str:
    """
    查询用户信息

    Args:
        phone: 手机号

    Returns:
        用户信息字符串
    """
    user = get_user_by_phone(phone)
    if not user:
        return "未查询到该手机号对应的用户信息"

    return f"""用户ID: {user.user_id}
用户名: {user.username}
手机号: {user.phone}
会员等级: {user.membership}
积分: {user.points}"""


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

    ticket_id = f"TK{datetime.now().strftime('%Y%m%d%H%m%S')}"
    return f"""已为您创建转接工单，工单号: {ticket_id}
转接原因: {reason}
请稍候，人工客服将尽快为您服务。
人工客服工作时间: 周一至周五 9:00-18:00
客服热线: 400-888-8888"""


from datetime import datetime


def get_all_tools():
    """获取所有工具"""
    return [query_order, query_logistics, query_user_info, transfer_to_human]
