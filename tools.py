import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


@tool
def query_order(order_id: str, user_id: Optional[str] = None) -> str:
    """Query order information by order ID."""
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
    """Query logistics information by order ID."""
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
    """Transfer conversation to human agent."""
    result = {
        "status": "success",
        "message": "Transferred to human agent. Please wait.",
        "ticket_id": "TK20240115001",
    }
    return str(result)
