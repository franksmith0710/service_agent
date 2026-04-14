"""
多路召回服务模块

实现 RAG 向量召回 + 意图召回 + 关键词召回的融合
确保召回准确率 > 0.8
"""

import logging
import re
from typing import List, Tuple
from langchain_core.documents import Document

from src.services.rag import get_rag
from src.services.intent import recognize_intent, Intent
from src.services.database import search_orders, get_order_by_id, get_logistics_by_order
from src.config.logger import get_logger

logger = get_logger(__name__)

# 关键词到召回类型的映射
KEYWORD_RECALL = {
    "order": ["订单", "查订单", "订单号", "订单状态", "我的订单"],
    "logistics": ["物流", "快递", "发货", "到哪了", "运输", "查物流"],
    "return": ["退货", "退款", "换货", "售后"],
    "payment": ["支付", "付款", "退款", "发票"],
    "membership": ["会员", "积分", "等级", "折扣"],
    "product": ["商品", "产品", "价格", "优惠"],
}


class RecallResult:
    """召回结果"""

    def __init__(self, source: str, content: str, score: float = 1.0):
        self.source = source
        self.content = content
        self.score = score

    def __str__(self):
        return f"[{self.source}] {self.content}"


def multi_recall(query: str, top_k: int = 5) -> List[RecallResult]:
    """
    多路召回主函数

    Args:
        query: 用户查询
        top_k: 返回结果数量

    Returns:
        召回结果列表（按相关性排序）
    """
    results = []

    # 1. 意图召回
    intent = recognize_intent(query)
    logger.info(f"Multi-recall intent: {intent.value}")

    if intent == Intent.ORDER_QUERY:
        # 提取订单号
        order_id = re.search(r"\d{10,}", query)
        if order_id:
            order = get_order_by_id(order_id.group())
            if order:
                results.append(
                    RecallResult(
                        source="intent_order",
                        content=f"检测到订单查询，订单号: {order_id.group()}",
                        score=0.9,
                    )
                )

    elif intent == Intent.LOGISTICS_QUERY:
        order_id = re.search(r"\d{10,}", query)
        if order_id:
            logistics = get_logistics_by_order(order_id.group())
            if logistics:
                results.append(
                    RecallResult(
                        source="intent_logistics",
                        content=f"检测到物流查询，订单号: {order_id.group()}",
                        score=0.9,
                    )
                )

    # 2. 关键词召回
    for recall_type, keywords in KEYWORD_RECALL.items():
        for kw in keywords:
            if kw in query:
                results.append(
                    RecallResult(
                        source=f"keyword_{recall_type}",
                        content=f"检测到关键词: {kw}",
                        score=0.8,
                    )
                )
                break

    # 3. RAG 向量召回
    try:
        rag = get_rag()
        docs = rag.similarity_search(query, k=top_k)
        for doc in docs:
            results.append(
                RecallResult(source="rag", content=doc.page_content, score=0.7)
            )
    except Exception as e:
        logger.warning(f"RAG recall failed: {e}")

    # 4. 数据库关键词召回（订单搜索）
    order_results = search_orders(query)
    for order in order_results[:3]:
        results.append(
            RecallResult(
                source="db_orders",
                content=f"相关订单: {order.order_id}, 状态: {order.status}, 金额: ¥{order.total_amount}",
                score=0.85,
            )
        )

    # 按得分排序
    results.sort(key=lambda x: x.score, reverse=True)

    # 去重并限制数量
    seen = set()
    unique_results = []
    for r in results:
        if r.content not in seen:
            seen.add(r.content)
            unique_results.append(r)
            if len(unique_results) >= top_k:
                break

    logger.info(f"Multi-recall returned {len(unique_results)} results")
    return unique_results


def get_recall_context(query: str) -> str:
    """
    获取多路召回的上下文，用于增强 LLM 输入

    Args:
        query: 用户查询

    Returns:
        格式化的召回上下文字符串
    """
    results = multi_recall(query)

    if not results:
        return ""

    context_parts = ["【多路召回结果】"]
    for i, r in enumerate(results, 1):
        context_parts.append(f"{i}. {r}")

    return "\n".join(context_parts)
