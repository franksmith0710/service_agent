"""
RAG 知识库服务模块

提供基于 Chroma 的向量知识库检索
"""

import os
import logging
from typing import Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config.settings import config
from src.config.logger import get_logger

logger = get_logger(__name__)

_rag_instance: Optional["KnowledgeBase"] = None


class KnowledgeBase:
    """知识库类"""

    def __init__(
        self,
        collection_name: str = "kefu_knowledge",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or config.chroma.persist_directory
        self.embeddings = self._create_embeddings()
        self.vector_store: Optional[Chroma] = None
        self._initialized = False

    def _create_embeddings(self) -> OllamaEmbeddings:
        """创建嵌入模型"""
        return OllamaEmbeddings(
            model=config.embedding.model,
            base_url=config.embedding.base_url,
        )

    def initialize(self) -> None:
        """初始化知识库"""
        if self._initialized:
            return

        os.makedirs(self.persist_directory, exist_ok=True)

        # 尝试加载已存在的 Chroma DB
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            # 验证 collection 是否有数据
            if self.vector_store._collection.count() > 0:
                logger.info(
                    f"Loaded existing knowledge base from {self.persist_directory}"
                )
            else:
                # collection 存在但无数据，重新填充
                logger.info("Collection exists but empty, populating with default docs")
                self._populate_default_docs()
        except Exception as e:
            logger.warning(f"Failed to load existing DB: {e}, creating new")
            self._populate_default_docs()

        self._initialized = True

    def _populate_default_docs(self) -> None:
        """填充默认文档"""
        docs = self._get_default_docs()
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        logger.info(f"Created new knowledge base with {len(docs)} documents")

    def _get_default_docs(self) -> list[Document]:
        """获取默认文档"""
        return [
            Document(
                page_content="我们的营业时间是周一至周五 9:00-18:00，节假日除外。",
                metadata={"category": "营业时间", "source": "basic_info"},
            ),
            Document(
                page_content="退换货政策：商品支持7天无理由退换货，质量问题我们承担运费。非质量问题运费由买家承担。",
                metadata={"category": "退换货", "source": "policy"},
            ),
            Document(
                page_content="支持支付宝、微信支付、银行卡等多种支付方式。企业买家可申请月结付款。",
                metadata={"category": "支付", "source": "payment"},
            ),
            Document(
                page_content="会员等级：普通会员（享受9.9折）、银卡会员（消费满1000，9.5折）、金卡会员（消费满5000，9折）、钻石会员（消费满10000，8.5折）。",
                metadata={"category": "会员", "source": "vip"},
            ),
            Document(
                page_content="订单问题请联系客服，提供订单号或收货手机号以便查询。",
                metadata={"category": "订单", "source": "faq"},
            ),
            Document(
                page_content="物流说明：默认顺丰速运，特殊商品走德邦或圆通。发货后1-3天送达，同城当日达。",
                metadata={"category": "物流", "source": "logistics"},
            ),
            Document(
                page_content="运费规则：满199元免运费，不足199元收取10元运费。偏远地区需额外加收20元。",
                metadata={"category": "运费", "source": "shipping"},
            ),
            Document(
                page_content="发票说明：支持普通发票和增值税专用发票。下单时备注发票抬头和税号，一般在收货后7个工作日内开具。",
                metadata={"category": "发票", "source": "invoice"},
            ),
            Document(
                page_content="售后服务：7天无理由退换货，30天质量问题换货，1年内质保维修。质保期外收取维修费用。",
                metadata={"category": "售后", "source": "service"},
            ),
            Document(
                page_content="联系我们：客服热线400-888-8888，邮箱support@example.com，工作时间周一至周五9:00-18:00。",
                metadata={"category": "联系", "source": "contact"},
            ),
        ]

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """获取检索器"""
        self.initialize()
        return self.vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 3})

    def add_documents(self, documents: list[Document]) -> None:
        """添加文档"""
        self.initialize()
        self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """
        相似性搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            文档列表
        """
        self.initialize()
        return self.vector_store.similarity_search(query, k=k)


def get_rag() -> KnowledgeBase:
    """获取 RAG 实例"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = KnowledgeBase()
    return _rag_instance
