"""
Agent 工具模块

提供订单查询、物流查询、用户信息查询、转人工客服等工具
集成完整的异常处理机制
"""

import json
from typing import Optional, Dict, Any
from langchain_core.tools import tool

from src.services.postgres import (
    get_order_by_id,
    get_order_by_phone,
    get_logistics_by_order,
    get_user_by_id,
    get_user_by_phone,
    search_orders,
    create_transfer_ticket,
)

from src.config.logger import get_logger

logger = get_logger(__name__)


def handle_tool_error(func_name: str, error: Exception) -> str:
    """工具错误处理"""
    logger.error(f"Tool {func_name} failed: {type(error).__name__}: {error}")
    return f"服务暂时不可用，请稍后重试。(错误: {type(error).__name__})"


@tool
def query_order(
    order_id: str = "", user_id: Optional[str] = None, phone: Optional[str] = None
) -> str:
    """
    查询订单信息。

    Args:
        order_id: 订单号（优先）
        user_id: 用户ID（可选）
        phone: 手机号（可选）

    Returns:
        订单信息字符串
    """
    try:
        if not order_id and phone:
            orders = get_order_by_phone(phone)
            if orders:
                order = orders[0]
                return _format_order(order)
            return "未查询到该手机号对应的订单"

        if not order_id:
            return "请提供订单号或手机号以便查询订单"

        order = get_order_by_id(order_id)
        if not order:
            orders = search_orders(order_id)
            if orders:
                return f"未找到订单号 {order_id}，为您找到以下相关订单：\n" + "\n".join(
                    [
                        f"- 订单号: {o['order_id']}, 状态: {o['status']}, 金额: ¥{o['total_amount']}"
                        for o in orders[:3]
                    ]
                )
            return f"未找到订单号 {order_id} 的订单信息"

        return _format_order(order)

    except Exception as e:
        return handle_tool_error("query_order", e)


def _format_order(order: Dict[str, Any]) -> str:
    """格式化订单信息"""
    items = order.get("items", [])
    if not items:
        items = [
            {
                "name": order.get("item_name", "商品"),
                "quantity": order.get("quantity", 1),
                "price": order.get("price", 0),
            }
        ]

    items_str = "\n".join(
        [
            f"  - {item.get('name', '商品')} x{item.get('quantity', 1)} ¥{item.get('price', 0)}"
            for item in items
        ]
    )

    return f"""订单号: {order.get("order_id", "N/A")}
状态: {order.get("status", "未知")}
商品:
{items_str or "无"}
总金额: ¥{order.get("total_amount", 0)}
下单时间: {order.get("created_at", "未知")}
支付方式: {order.get("pay_method", "未知")}
收货地址: {order.get("shipping_address", "未填写")}"""


@tool
def query_logistics(order_id: str = "", phone: Optional[str] = None) -> str:
    """
    查询物流信息。

    Args:
        order_id: 订单号（优先）
        phone: 手机号（可选）

    Returns:
        物流信息字符串
    """
    try:
        if not order_id and phone:
            orders = get_order_by_phone(phone)
            if orders:
                for o in orders:
                    if o.get("status") in ["已发货", "配送中"]:
                        order_id = o.get("order_id")
                        break
                if not order_id:
                    return "该手机号下没有已发货的订单"
            else:
                return "未查询到该手机号对应的订单"

        if not order_id:
            return "请提供订单号以便查询物流"

        logistics = get_logistics_by_order(order_id)
        if not logistics:
            order = get_order_by_id(order_id)
            if not order:
                return f"未找到订单号 {order_id} 的信息"
            if order.get("status") in ["待发货", "已取消"]:
                return f"订单号 {order_id} 尚未发货，当前状态: {order.get('status')}"
            return f"订单号 {order_id} 暂无物流信息"

        return _format_logistics(logistics)

    except Exception as e:
        return handle_tool_error("query_logistics", e)


def _format_logistics(logistics: Dict[str, Any]) -> str:
    """格式化物流信息"""
    trace = logistics.get("trace", [])
    if isinstance(trace, str):
        try:
            trace = json.loads(trace)
        except:
            trace = []

    trace_str = "\n".join(
        [
            f"  [{t.get('time', '未知')}] {t.get('location', '未知')} - {t.get('status', '未知')}"
            for t in trace
        ]
    )

    return f"""订单号: {logistics.get("order_id", "N/A")}
承运商: {logistics.get("carrier", "未知")}
运单号: {logistics.get("tracking_number", "未知")}
当前状态: {logistics.get("status", "未知")}

物流轨迹:
{trace_str or "暂无轨迹"}"""


@tool
def query_user_info(phone: str) -> str:
    """
    查询用户信息。

    Args:
        phone: 手机号

    Returns:
        用户信息字符串
    """
    try:
        user = get_user_by_phone(phone)
        if not user:
            return "未查询到该手机号对应的用户信息"

        return f"""用户ID: {user.get("user_id", "N/A")}
用户名: {user.get("username", "未知")}
手机号: {user.get("phone", "未知")}
会员等级: {user.get("membership", "普通会员")}
积分: {user.get("points", 0)}"""

    except Exception as e:
        return handle_tool_error("query_user_info", e)


@tool
def transfer_to_human(
    reason: str,
    conversation_summary: Optional[str] = None,
    user_id: Optional[str] = None,
    phone: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    转接人工客服。

    Args:
        reason: 转接原因
        conversation_summary: 对话摘要（可选）
        user_id: 用户ID（可选）
        phone: 手机号（可选）
        session_id: 会话ID（可选）

    Returns:
        转接结果字符串
    """
    try:
        result = create_transfer_ticket(
            user_id=user_id,
            phone=phone,
            session_id=session_id or "",
            reason=reason,
            summary=conversation_summary,
        )

        ticket_id = result["ticket_id"]
        logger.info(f"Transfer to human: {ticket_id}")

        return f"""已为您创建转接工单，工单号: {ticket_id}
转接原因: {reason}
请稍候，人工客服将尽快为您服务。
人工客服工作时间: 周一至周五 9:00-18:00
客服热线: 400-990-5898"""

    except Exception as e:
        return handle_tool_error("transfer_to_human", e)


def get_all_tools():
    """获取所有工具"""
    return [query_order, query_logistics, query_user_info, transfer_to_human]
