"""
插入用户和订单数据
"""

import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="123456",
    database="kefu_agent",
    client_encoding="UTF8",
)
cursor = conn.cursor()

# Users
users = [
    ("U001", "张三", "13800138000", "黄金会员", 5000),
    ("U002", "李四", "13800138001", "白银会员", 2000),
    ("U003", "王五", "13800138002", "铂金会员", 10000),
    ("U004", "赵六", "13800138003", "普通会员", 500),
    ("U005", "钱七", "13800138004", "钻石会员", 20000),
]
for u in users:
    cursor.execute(
        "INSERT INTO users (user_id, username, phone, membership, points) VALUES (%s, %s, %s, %s, %s)",
        u,
    )

# Orders
orders = [
    (
        "20250413001",
        "U001",
        "已发货",
        "蛟龙16 Pro",
        1,
        7999,
        7999,
        "在线支付",
        "北京市朝阳区",
    ),
    (
        "20250413002",
        "U001",
        "待发货",
        "极光Pro",
        1,
        6999,
        6999,
        "在线支付",
        "北京市海淀区",
    ),
    (
        "20250413003",
        "U002",
        "已完成",
        "无界16 Pro",
        1,
        5499,
        5499,
        "支付宝",
        "上海市浦东区",
    ),
    (
        "20250414001",
        "U003",
        "已发货",
        "旷世16 Pro",
        1,
        10999,
        10999,
        "在线支付",
        "广州市天河区",
    ),
    (
        "20250414002",
        "U004",
        "待发货",
        "翼龙16",
        1,
        7499,
        7499,
        "微信支付",
        "深圳市南山区",
    ),
]
for o in orders:
    cursor.execute(
        "INSERT INTO orders (order_id, user_id, status, item_name, quantity, price, total_amount, pay_method, shipping_address) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        o,
    )

# Logistics
logistics = [
    ("20250413001", "顺丰", "SF123456789", "配送中", "北京市"),
    ("20250414001", "圆通", "YTO234567890", "配送中", "广州市"),
]
for l in logistics:
    cursor.execute(
        "INSERT INTO logistics (order_id, carrier, tracking_number, status, current_location) VALUES (%s, %s, %s, %s, %s)",
        l,
    )

conn.commit()
print(f"Inserted {len(users)} users, {len(orders)} orders, {len(logistics)} logistics")
cursor.close()
conn.close()
