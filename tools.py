from typing import Optional
from pydantic import BaseModel, Field


class OrderQueryInput(BaseModel):
    order_id: str = Field(description="订单号")
    user_id: Optional[str] = Field(default=None, description="用户ID")


class OrderInfo(BaseModel):
    order_id: str
    status: str
    items: list[str]
    total_amount: float
    created_at: str


def query_order(order_id: str, user_id: Optional[str] = None) -> str:
    """
    查询订单信息。

    Args:
        order_id: 订单号
        user_id: 用户ID（可选）

    Returns:
        订单信息JSON字符串
    """
    # 占位符 - 后续替换为真实数据库查询
    order_info = {
        "order_id": order_id,
        "status": "已发货",
        "items": ["商品A x1", "商品B x2"],
        "total_amount": 299.00,
        "created_at": "2024-01-15 10:30:00",
    }
    return str(order_info)


class LogisticsQueryInput(BaseModel):
    order_id: str = Field(description="订单号")


class LogisticsInfo(BaseModel):
    order_id: str
    carrier: str
    tracking_number: str
    status: str
    trace: list[dict]


def query_logistics(order_id: str) -> str:
    """
    查询物流信息。

    Args:
        order_id: 订单号

    Returns:
        物流信息JSON字符串
    """
    # 占位符 - 后续替换为真实API查询
    logistics_info = {
        "order_id": order_id,
        "carrier": "顺丰速运",
        "tracking_number": "SF1234567890",
        "status": "运输中",
        "trace": [
            {
                "time": "2024-01-16 14:00",
                "location": "上海分拨中心",
                "status": "已发出",
            },
            {"time": "2024-01-15 20:00", "location": "北京仓库", "status": "已揽收"},
        ],
    }
    return str(logistics_info)


class TransferToHumanInput(BaseModel):
    reason: str = Field(description="转接原因")
    conversation_summary: Optional[str] = Field(default=None, description="对话摘要")


def transfer_to_human(reason: str, conversation_summary: Optional[str] = None) -> str:
    """
    将对话转接给人工客服。

    Args:
        reason: 转接原因
        conversation_summary: 对话摘要（可选）

    Returns:
        转接结果
    """
    # 占位符 - 后续替换为真实转接逻辑
    result = {
        "status": "success",
        "message": "已为您转接人工客服，请稍候...",
        "ticket_id": "TK20240115001",
    }
    return str(result)


TOOLS = [
    {
        "name": "query_order",
        "description": "查询订单信息，包括订单状态、商品列表、金额等",
        "parameters": OrderQueryInput.model_json_schema(),
    },
    {
        "name": "query_logistics",
        "description": "查询物流信息，包括快递公司、运单号、运输轨迹等",
        "parameters": LogisticsQueryInput.model_json_schema(),
    },
    {
        "name": "transfer_to_human",
        "description": "当用户需要人工服务时，将对话转接给人工客服",
        "parameters": TransferToHumanInput.model_json_schema(),
    },
]
