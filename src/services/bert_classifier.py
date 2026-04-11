"""
BERT 意图分类器服务

基于 sentence-transformers 的语义相似度分类器
作为规则匹配的兜底方案
"""

import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.config.logger import get_logger

logger = get_logger(__name__)

# 意图类别描述（用于语义匹配）
INTENT_DESCRIPTORS = {
    "query_order": "查询订单、订单号、订单状态、订单详情",
    "query_logistics": "查询物流、快递、发货、运输、到哪了",
    "transfer": "转人工、转接客服、投诉、找真人",
    "chat": "一般聊天、商品咨询、常见问题解答",
}

_model: Optional["BERTIntentClassifier"] = None


class BERTIntentClassifier:
    """BERT 意图分类器"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化分类器

        Args:
            model_name: SentenceTransformer 模型名称
        """
        logger.info(f"Loading BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.labels = list(INTENT_DESCRIPTORS.keys())
        # 预计算类别描述的嵌入
        self.label_embeddings = self.model.encode(
            [INTENT_DESCRIPTORS[label] for label in self.labels]
        )
        logger.info("BERT classifier initialized")

    def predict(self, text: str, threshold: float = 0.5) -> str:
        """
        预测意图

        Args:
            text: 用户输入文本
            threshold: 相似度阈值

        Returns:
            预测的意图标签
        """
        # 编码输入文本
        text_embedding = self.model.encode([text])

        # 计算与各类别的相似度
        similarities = cosine_similarity(text_embedding, self.label_embeddings)[0]

        # 获取最高相似度
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score < threshold:
            logger.warning(f"Low confidence: {best_score:.2f}, fallback to chat")
            return "chat"

        predicted_intent = self.labels[best_idx]
        logger.info(f"BERT prediction: {predicted_intent} (score: {best_score:.2f})")

        return predicted_intent


def get_classifier() -> BERTIntentClassifier:
    """获取分类器实例（单例）"""
    global _model
    if _model is None:
        _model = BERTIntentClassifier()
    return _model


def predict_intent(text: str) -> str:
    """
    预测意图（便捷函数）

    Args:
        text: 用户输入文本

    Returns:
        预测的意图标签
    """
    classifier = get_classifier()
    return classifier.predict(text)
