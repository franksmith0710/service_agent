"""工具函数模块 - 客服Agent可调用的工具"""
import json
from typing import Dict, Any, List
from langchain_core.tools import tool


@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库，获取与用户问题相关的 FAQ 和帮助信息。
    
    Args:
        query: 用户的查询内容
    
    Returns:
        知识库中匹配的相关信息，如果没找到返回空字符串
    """
    from agent import rag_retriever
    
    if rag_retriever is None:
        return "知识库暂不可用，请稍后再试"
    
    try:
        docs = rag_retriever.invoke(query)
        if docs:
            results = "\n\n".join([doc.page_content for doc in docs[:3]])
            return results if results else "未找到相关信息"
        return "未找到相关信息"
    except Exception as e:
        return f"查询知识库时出错: str(e)"


@tool
def query_order(order_id: str) -> str:
    """查询订单信息。
    
    Args:
        order_id: 订单号
    
    Returns:
        订单信息 JSON 字符串，包含订单状态、金额、商品等信息
    """
    # TODO: 接入真实订单系统
    # 这里模拟返回订单信息
    mock_orders = {
        "ORD001": {"status": "已发货", "amount": "299.00元", "items": "蓝牙耳机", "create_time": "2026-04-01"},
        "ORD002": {"status": "待支付", "amount": "599.00元", "items": "智能手表", "create_time": "2026-04-05"},
    }
    
    if order_id in mock_orders:
        order = mock_orders[order_id]
        return json.dumps({
            "订单号": order_id,
            "订单状态": order["status"],
            "商品": order["items"],
            "金额": order["amount"],
            "下单时间": order["create_time"]
        }, ensure_ascii=False)
    else:
        return json.dumps({"error": f"未找到订单号 {order_id} 的订单信息"}, ensure_ascii=False)


@tool
def query_logistics(tracking_number: str) -> str:
    """查询物流信息。
    
    Args:
        tracking_number: 快递单号
    
    Returns:
        物流信息 JSON 字符串，包含快递公司、当前状态、物流轨迹等
    """
    # TODO: 接入真实物流系统
    mock_logistics = {
        "SF123456789": {
            "company": "顺丰速运",
            "status": "运输中",
            "current": "杭州分拨中心",
            "tracks": [
                {"time": "2026-04-07 14:30", "location": "杭州分拨中心", "status": "已发出"},
                {"time": "2026-04-07 10:00", "location": "杭州仓", "status": "已打包"},
                {"time": "2026-04-06 20:00", "location": "杭州仓", "status": "已下单"}
            ]
        },
        "YT987654321": {
            "company": "圆通速递",
            "status": "已签收",
            "current": "上海市浦东新区",
            "tracks": [
                {"time": "2026-04-07 09:00", "location": "上海浦东分部", "status": "已签收"},
                {"time": "2026-04-06 18:00", "location": "上海分拨中心", "status": "运输中"},
                {"time": "2026-04-05 10:00", "location": "杭州仓", "status": "已发出"}
            ]
        }
    }
    
    if tracking_number in mock_logistics:
        logistics = mock_logistics[tracking_number]
        tracks_str = "\n".join([
            f"  {t['time']} - {t['location']} - {t['status']}"
            for t in logistics['tracks']
        ])
        return json.dumps({
            "快递单号": tracking_number,
            "快递公司": logistics['company'],
            "当前状态": logistics['status'],
            "当前位置": logistics['current'],
            "物流轨迹": tracks_str
        }, ensure_ascii=False, indent=2)
    else:
        return json.dumps({"error": f"未找到快递单号 {tracking_number} 的物流信息"}, ensure_ascii=False)


@tool
def transfer_to_human(reason: str) -> str:
    """转接人工客服。当需要人工处理时调用此工具。
    
    Args:
        reason: 转接原因，说明为什么需要人工处理
    
    Returns:
        转接结果信息
    """
    # TODO: 接入真实人工客服系统
    return json.dumps({
        "success": True,
        "message": "已为您转接人工客服，请稍候...",
        "转接原因": reason,
        "预计等待时间": "3-5分钟"
    }, ensure_ascii=False)


# 工具列表，供 Agent 使用
AVAILABLE_TOOLS = [search_knowledge_base, query_order, query_logistics, transfer_to_human]