"""
模拟数据库模块

提供订单、物流、用户等业务数据的模拟存储
用于 Demo 演示，后续可替换为真实数据库
"""

from typing import Optional, List
from datetime import datetime, timedelta
import random


class Order:
    """订单模型"""

    def __init__(
        self,
        order_id: str,
        user_id: str,
        status: str,
        items: List[dict],
        total_amount: float,
        created_at: str,
        pay_method: str = "微信支付",
        shipping_address: str = "",
    ):
        self.order_id = order_id
        self.user_id = user_id
        self.status = status
        self.items = items
        self.total_amount = total_amount
        self.created_at = created_at
        self.pay_method = pay_method
        self.shipping_address = shipping_address

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "user_id": self.user_id,
            "status": self.status,
            "items": self.items,
            "total_amount": self.total_amount,
            "created_at": self.created_at,
            "pay_method": self.pay_method,
            "shipping_address": self.shipping_address,
        }


class Logistics:
    """物流模型"""

    def __init__(
        self,
        order_id: str,
        carrier: str,
        tracking_number: str,
        status: str,
        trace: List[dict],
    ):
        self.order_id = order_id
        self.carrier = carrier
        self.tracking_number = tracking_number
        self.status = status
        self.trace = trace

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "carrier": self.carrier,
            "tracking_number": self.tracking_number,
            "status": self.status,
            "trace": self.trace,
        }


class User:
    """用户模型"""

    def __init__(
        self,
        user_id: str,
        username: str,
        phone: str,
        membership: str = "普通会员",
        points: int = 0,
    ):
        self.user_id = user_id
        self.username = username
        self.phone = phone
        self.membership = membership
        self.points = points

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "phone": self.phone,
            "membership": self.membership,
            "points": self.points,
        }


# ==================== 模拟数据 ====================

# 模拟用户数据
USERS = {
    "U001": User("U001", "张三", "13800138001", "金卡会员", 5500),
    "U002": User("U002", "李四", "13800138002", "银卡会员", 1200),
    "U003": User("U003", "王五", "13800138003", "钻石会员", 15000),
    "U004": User("U004", "赵六", "13800138004", "普通会员", 300),
}

# 模拟订单数据
ORDERS = {
    "1234567890": Order(
        order_id="1234567890",
        user_id="U001",
        status="已发货",
        items=[
            {"name": "iPhone 15 Pro", "quantity": 1, "price": 7999.00},
            {"name": "手机壳", "quantity": 1, "price": 99.00},
        ],
        total_amount=8098.00,
        created_at="2024-01-15 10:30:00",
        pay_method="微信支付",
        shipping_address="北京市朝阳区xxx街道xxx小区",
    ),
    "2234567891": Order(
        order_id="2234567891",
        user_id="U001",
        status="已完成",
        items=[
            {"name": "MacBook Air M3", "quantity": 1, "price": 9499.00},
        ],
        total_amount=9499.00,
        created_at="2024-01-10 15:20:00",
        pay_method="支付宝",
        shipping_address="北京市朝阳区xxx街道xxx小区",
    ),
    "3234567892": Order(
        order_id="3234567892",
        user_id="U002",
        status="待发货",
        items=[
            {"name": "AirPods Pro 2", "quantity": 2, "price": 1899.00},
        ],
        total_amount=3798.00,
        created_at="2024-01-18 09:15:00",
        pay_method="微信支付",
        shipping_address="上海市浦东新区xxx路xxx号",
    ),
    "4234567893": Order(
        order_id="4234567893",
        user_id="U003",
        status="已发货",
        items=[
            {"name": "iPad Pro 12.9", "quantity": 1, "price": 9999.00},
            {"name": "Apple Pencil", "quantity": 1, "price": 999.00},
        ],
        total_amount=10998.00,
        created_at="2024-01-20 14:00:00",
        pay_method="银行卡",
        shipping_address="广州市天河区xxx大道xxx号",
    ),
    "5234567894": Order(
        order_id="5234567894",
        user_id="U001",
        status="配送中",
        items=[
            {"name": "Apple Watch Ultra 2", "quantity": 1, "price": 6999.00},
        ],
        total_amount=6999.00,
        created_at="2024-01-22 11:30:00",
        pay_method="微信支付",
        shipping_address="北京市朝阳区xxx街道xxx小区",
    ),
}

# 模拟物流数据
LOGISTICS = {
    "1234567890": Logistics(
        order_id="1234567890",
        carrier="顺丰速运",
        tracking_number="SF1234567890",
        status="运输中",
        trace=[
            {
                "time": "2024-01-16 14:00",
                "location": "上海分拨中心",
                "status": "已发出",
            },
            {"time": "2024-01-15 20:00", "location": "北京仓库", "status": "已发货"},
            {"time": "2024-01-15 10:30", "location": "系统", "status": "已下单"},
        ],
    ),
    "4234567893": Logistics(
        order_id="4234567893",
        carrier="京东物流",
        tracking_number="JD4234567893",
        status="已签收",
        trace=[
            {
                "time": "2024-01-22 09:30",
                "location": "广州市天河区",
                "status": "已签收",
            },
            {
                "time": "2024-01-21 18:00",
                "location": "广州天河营业部",
                "status": "派送中",
            },
            {
                "time": "2024-01-21 07:00",
                "location": "广州分拨中心",
                "status": "运输中",
            },
            {
                "time": "2024-01-20 20:00",
                "location": "深圳分拨中心",
                "status": "已发出",
            },
        ],
    ),
    "5234567894": Logistics(
        order_id="5234567894",
        carrier="顺丰速运",
        tracking_number="SF5234567894",
        status="派送中",
        trace=[
            {"time": "2024-01-23 08:00", "location": "北京朝阳区", "status": "派送中"},
            {
                "time": "2024-01-22 22:00",
                "location": "北京分拨中心",
                "status": "已到达",
            },
            {
                "time": "2024-01-22 14:00",
                "location": "石家庄中转站",
                "status": "运输中",
            },
            {"time": "2024-01-22 11:30", "location": "系统", "status": "已发货"},
        ],
    ),
}


# ==================== 数据库操作 ====================


def get_order_by_id(order_id: str) -> Optional[Order]:
    """根据订单号查询订单"""
    return ORDERS.get(order_id)


def get_order_by_phone(phone: str) -> List[Order]:
    """根据手机号查询订单"""
    user = None
    for u in USERS.values():
        if u.phone == phone:
            user = u
            break
    if not user:
        return []
    return [o for o in ORDERS.values() if o.user_id == user.user_id]


def get_logistics_by_order(order_id: str) -> Optional[Logistics]:
    """根据订单号查询物流"""
    return LOGISTICS.get(order_id)


def get_user_by_id(user_id: str) -> Optional[User]:
    """根据用户ID查询用户"""
    return USERS.get(user_id)


def get_user_by_phone(phone: str) -> Optional[User]:
    """根据手机号查询用户"""
    for u in USERS.values():
        if u.phone == phone:
            return u
    return None


def search_orders(keyword: str) -> List[Order]:
    """搜索订单（订单号、商品名称）"""
    results = []
    keyword = keyword.lower()
    for order in ORDERS.values():
        if keyword in order.order_id:
            results.append(order)
        else:
            for item in order.items:
                if keyword in item["name"].lower():
                    results.append(order)
                    break
    return results
