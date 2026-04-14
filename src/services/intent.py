"""
意图识别模块

基于规则的意图识别 + embedding 语义匹配兜底
"""

import re
import os
import numpy as np
from enum import Enum
from typing import Optional, List
from langchain_ollama import OllamaEmbeddings

from src.config.settings import config
from src.config.logger import get_logger

logger = get_logger(__name__)


class Intent(Enum):
    """意图枚举"""

    ORDER_QUERY = "query_order"  # 订单查询
    LOGISTICS_QUERY = "query_logistics"  # 物流查询
    USER_QUERY = "query_user_info"  # 用户信息查询
    TRANSFER_HUMAN = "transfer"  # 转人工
    GENERAL_CHAT = "chat"  # 一般聊天
    PRODUCT_QUERY = "query_product"  # 产品咨询
    UNKNOWN = "unknown"  # 未知


INTENT_PATTERNS = {
    Intent.ORDER_QUERY: [
        r"订单[号码]?[是]?\d+",
        r"查.{0,3}订单",
        r"订单状态",
        r"订单详情",
        r"订单信息",
        r"我的订单",
        r"订单\d{10,}",
        r"(?:order|订单).*?(?:查询|号|状态)",
    ],
    Intent.LOGISTICS_QUERY: [
        r"物流[信息]?",
        r"快递[信息]?",
        r"运输[信息]?",
        r"发货",
        r"到哪[里了]?",
        r"物流状态",
        r"快递单号",
        r"查.{0,3}物流",
        r"(?:logistics|快递).*?(?:查询|单号)",
    ],
    Intent.USER_QUERY: [
        r"用户[信息]?",
        r"会员[信息]?",
        r"账号信息",
        r"查.{0,3}用户",
        r"我的资料",
        r"个人信息",
    ],
    Intent.TRANSFER_HUMAN: [
        r"转人工",
        r"转接人工",
        r"人工客服",
        r"客服电话",
        r"我要投诉",
        r"找真人",
        r"human.*service",
        r"transfer.*human",
    ],
}

INTENT_EXAMPLES = {
    Intent.ORDER_QUERY: [
        "查一下我的订单",
        "订单到哪了",
        "我想看订单状态",
        "订单号是多少",
        "看看我买的电脑到哪了",
        "帮我查订单",
    ],
    Intent.LOGISTICS_QUERY: [
        "物流信息",
        "快递到哪了",
        "发货了吗",
        "什么时候到",
        "查一下物流",
        "我的快递到哪了",
    ],
    Intent.USER_QUERY: [
        "查一下我的信息",
        "我的会员等级",
        "我有多少积分",
        "用户信息",
        "我的账号",
    ],
    Intent.PRODUCT_QUERY: [
        "蛟龙16 Pro怎么样",
        "极光Pro配置",
        "耀世16 Pro价格",
        "无界16 Pro续航",
        "推荐一款游戏本",
    ],
    Intent.TRANSFER_HUMAN: [
        "我要转人工",
        "找客服",
        "投诉",
        "找真人客服",
    ],
}

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OllamaEmbeddings(
            model=config.embedding.model,
            base_url=config.embedding.base_url,
        )
    return _embedding_model


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _semantic_match_intent(text: str) -> Intent:
    """基于 embedding 的语义匹配"""
    try:
        emb_model = _get_embedding_model()
        text_emb = emb_model.embed_query(text)

        best_intent = Intent.GENERAL_CHAT
        best_score = 0.0

        for intent, examples in INTENT_EXAMPLES.items():
            example_embs = emb_model.embed_documents(examples)
            scores = [_cosine_similarity(text_emb, ex_emb) for ex_emb in example_embs]
            avg_score = sum(scores) / len(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_intent = intent

        if best_score < 0.5:
            return Intent.GENERAL_CHAT
        return best_intent

    except Exception as e:
        logger.warning(f"Semantic match failed: {e}")
        return Intent.GENERAL_CHAT


def recognize_intent(text: str) -> Intent:
    """
    识别用户意图（规则 + 语义匹配兜底）

    Args:
        text: 用户输入文本

    Returns:
        识别的意图
    """
    text_lower = text.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return intent

    return _semantic_match_intent(text)


def extract_order_id(text: str) -> Optional[str]:
    """
    从文本中提取订单号

    Args:
        text: 用户输入文本

    Returns:
        订单号，如果未找到返回 None
    """
    # 优先匹配10位以上数字
    match = re.search(r"\d{10,}", text)
    if match:
        return match.group()

    # 尝试匹配其他格式
    match = re.search(r"(?:订单号?|order)[：:]\s*(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def extract_phone(text: str) -> Optional[str]:
    """
    从文本中提取手机号

    Args:
        text: 用户输入文本

    Returns:
        手机号，如果未找到返回 None
    """
    # 匹配11位手机号
    match = re.search(r"1[3-9]\d{9}", text)
    if match:
        return match.group()

    # 尝试匹配其他格式
    match = re.search(r"(?:手机|电话|号码)[：:]\s*(1[3-9]\d{9})", text)
    if match:
        return match.group(1)

    return None


def should_use_tools(text: str) -> bool:
    """
    判断是否需要使用工具

    Args:
        text: 用户输入文本

    Returns:
        是否需要使用工具
    """
    intent = recognize_intent(text)
    return intent in [Intent.ORDER_QUERY, Intent.LOGISTICS_QUERY, Intent.TRANSFER_HUMAN]


def get_intent_description(text: str) -> str:
    """
    获取意图描述（用于调试）

    Args:
        text: 用户输入文本

    Returns:
        意图描述
    """
    intent = recognize_intent(text)
    return intent.value
