"""
PostgreSQL 数据库层

提供数据库连接（线程安全连接池）和规范化数据查询接口
"""

import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from src.config.settings import config
from src.config.logger import get_logger

logger = get_logger(__name__)

_connection_pool = None
MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 10


def get_connection_pool():
    """获取连接池（线程安全）"""
    global _connection_pool

    if _connection_pool is None:
        try:
            _connection_pool = ThreadedConnectionPool(
                minconn=MIN_CONNECTIONS,
                maxconn=MAX_CONNECTIONS,
                host=config.postgres.host,
                port=config.postgres.port,
                user=config.postgres.user,
                password=config.postgres.password,
                database=config.postgres.database,
                client_encoding="UTF8",
                connect_timeout=5,
            )
            logger.info(
                f"PostgreSQL pool created: {MIN_CONNECTIONS}-{MAX_CONNECTIONS} connections"
            )
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL pool creation failed: {e}")
            raise

    return _connection_pool


def get_connection():
    """从连接池获取连接"""
    pool = get_connection_pool()
    try:
        conn = pool.getconn()
        conn.autocommit = False
        return conn
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {e}")
        raise


def return_connection(conn):
    """归还连接到池"""
    if conn and _connection_pool:
        try:
            _connection_pool.putconn(conn)
        except Exception as e:
            logger.warning(f"Failed to return connection to pool: {e}")


@contextmanager
def get_cursor():
    """获取数据库游标的上下文管理器"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        yield cursor
        conn.commit()
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            return_connection(conn)


def execute_query(
    query: str, params: tuple = None, timeout: int = 10
) -> List[Dict[str, Any]]:
    """执行查询 SQL"""
    with get_cursor() as cursor:
        if timeout:
            cursor.execute(f"SET statement_timeout = {timeout * 1000}")
        cursor.execute(query, params)
        results = cursor.fetchall()
        return [dict(row) for row in results]


def execute_update(query: str, params: tuple = None, timeout: int = 10) -> int:
    """执行更新 SQL"""
    with get_cursor() as cursor:
        if timeout:
            cursor.execute(f"SET statement_timeout = {timeout * 1000}")
        cursor.execute(query, params)
        return cursor.rowcount


def test_connection() -> bool:
    """测试数据库连接"""
    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def _to_float(value) -> Optional[float]:
    """转换值为浮点数"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_datetime(dt) -> Optional[str]:
    """格式化时间"""
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


def get_user_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    """根据手机号查询用户"""
    results = execute_query("SELECT * FROM users WHERE phone = %s", (phone,))
    if not results:
        return None
    r = results[0]
    return {
        "user_id": r["user_id"],
        "username": r["username"],
        "phone": r["phone"],
        "membership": r.get("membership"),
        "points": r.get("points", 0),
    }


def get_order_by_id(order_id: str) -> Optional[Dict[str, Any]]:
    """根据订单号查询订单（支持多商品）"""
    results = execute_query("SELECT * FROM orders WHERE order_id = %s", (order_id,))
    if not results:
        return None

    first_r = results[0]
    items = [
        {
            "name": r["item_name"],
            "quantity": r["quantity"],
            "price": _to_float(r["price"]),
        }
        for r in results
    ]

    return {
        "order_id": first_r["order_id"],
        "user_id": first_r["user_id"],
        "status": first_r["status"],
        "items": items,
        "total_amount": _to_float(first_r["total_amount"]),
        "created_at": _format_datetime(first_r.get("created_at")),
        "pay_method": first_r.get("pay_method"),
        "shipping_address": first_r.get("shipping_address"),
    }


def get_order_by_phone(phone: str) -> List[Dict[str, Any]]:
    """根据手机号查询订单"""
    results = execute_query(
        """SELECT o.* FROM orders o
           JOIN users u ON o.user_id = u.user_id
           WHERE u.phone = %s
           ORDER BY o.created_at DESC""",
        (phone,),
    )
    return [_format_order(r) for r in results]


def _format_order(r: Dict[str, Any]) -> Dict[str, Any]:
    """格式化订单数据（支持多商品）"""
    return {
        "order_id": r["order_id"],
        "user_id": r["user_id"],
        "status": r["status"],
        "items": [
            {
                "name": r["item_name"],
                "quantity": r["quantity"],
                "price": _to_float(r["price"]),
            }
        ],
        "total_amount": _to_float(r["total_amount"]),
        "created_at": _format_datetime(r.get("created_at")),
        "pay_method": r.get("pay_method"),
        "shipping_address": r.get("shipping_address"),
    }


def search_orders(keyword: str) -> List[Dict[str, Any]]:
    """搜索订单"""
    results = execute_query(
        """SELECT * FROM orders
           WHERE order_id LIKE %s OR item_name LIKE %s
           ORDER BY created_at DESC""",
        (f"%{keyword}%", f"%{keyword}%"),
    )
    return [_format_order(r) for r in results]


def get_logistics_by_order(order_id: str) -> Optional[Dict[str, Any]]:
    """根据订单号查询物流"""
    results = execute_query("SELECT * FROM logistics WHERE order_id = %s", (order_id,))
    if not results:
        return None
    r = results[0]
    trace = r.get("trace")
    if isinstance(trace, str):
        import json

        try:
            trace = json.loads(trace)
        except:
            trace = []
    return {
        "order_id": r["order_id"],
        "carrier": r["carrier"],
        "tracking_number": r["tracking_number"],
        "status": r["status"],
        "current_location": r.get("current_location"),
        "trace": trace or [],
    }


def create_transfer_ticket(
    user_id: Optional[str],
    phone: Optional[str],
    session_id: str,
    reason: str,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    """创建转人工工单"""
    import time
    ticket_id = f"TK{int(time.time() * 1000)}"

    execute_update(
        """INSERT INTO transfer_tickets 
           (ticket_id, user_id, phone, session_id, reason, summary, status, created_at, updated_at)
           VALUES (%s, %s, %s, %s, %s, %s, 'pending', NOW(), NOW())""",
        (ticket_id, user_id, phone, session_id, reason, summary),
    )

    logger.info(f"Created transfer ticket: {ticket_id}")
    return {"ticket_id": ticket_id, "status": "pending"}


def get_ticket_by_id(ticket_id: str) -> Optional[Dict[str, Any]]:
    """根据工单号查询工单"""
    results = execute_query(
        "SELECT * FROM transfer_tickets WHERE ticket_id = %s", (ticket_id,)
    )
    if not results:
        return None
    return _format_ticket(results[0])


def get_tickets_by_status(
    status: str = "pending", limit: int = 50
) -> List[Dict[str, Any]]:
    """根据状态查询工单列表"""
    results = execute_query(
        """SELECT * FROM transfer_tickets 
           WHERE status = %s 
           ORDER BY created_at DESC 
           LIMIT %s""",
        (status, limit),
    )
    return [_format_ticket(r) for r in results]


def get_all_tickets(limit: int = 50) -> List[Dict[str, Any]]:
    """查询所有工单"""
    results = execute_query(
        """SELECT * FROM transfer_tickets 
           ORDER BY created_at DESC 
           LIMIT %s""",
        (limit,),
    )
    return [_format_ticket(r) for r in results]


def update_ticket_status(ticket_id: str, status: str) -> bool:
    """更新工单状态"""
    valid_statuses = ["pending", "processing", "resolved"]
    if status not in valid_statuses:
        logger.warning(f"Invalid status: {status}")
        return False

    rowcount = execute_update(
        """UPDATE transfer_tickets 
           SET status = %s, updated_at = NOW() 
           WHERE ticket_id = %s""",
        (status, ticket_id),
    )
    return rowcount > 0


def _format_ticket(r: Dict[str, Any]) -> Dict[str, Any]:
    """格式化工单数据"""
    return {
        "ticket_id": r["ticket_id"],
        "user_id": r.get("user_id"),
        "phone": r.get("phone"),
        "session_id": r.get("session_id"),
        "reason": r["reason"],
        "summary": r.get("summary"),
        "status": r["status"],
        "created_at": _format_datetime(r.get("created_at")),
        "updated_at": _format_datetime(r.get("updated_at")),
    }
