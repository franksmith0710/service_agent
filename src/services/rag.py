"""
RAG 知识库服务模块

提供基于 Chroma 的向量知识库检索
支持从 data/kb_*.txt 加载知识库数据
"""

import os
import re
import logging
import hashlib
from typing import Optional, List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config.settings import config
from src.config.logger import get_logger

logger = get_logger(__name__)

_rag_instance: Optional["KnowledgeBase"] = None


def _compute_content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_kb_files(data_dir: str = "./data") -> List[Document]:
    """从 data/kb_*.txt 加载知识库文档

    Args:
        data_dir: 数据文件目录

    Returns:
        Document 列表
    """
    documents = []

    # 加载 brand
    brand_path = os.path.join(data_dir, "kb_brand.txt")
    if os.path.exists(brand_path):
        with open(brand_path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.strip():
            doc = Document(
                page_content=content,
                metadata={"source": "kb_brand.txt", "type": "brand"},
            )
            documents.append(doc)
            logger.info(f"Loaded: kb_brand.txt")

    # 加载 products（解析每个产品）
    products_path = os.path.join(data_dir, "kb_products.txt")
    if os.path.exists(products_path):
        with open(products_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按 ## 产品 分割
        product_blocks = re.split(r"(?=## 产品\d+)", content)
        for block in product_blocks:
            if not block.strip() or not block.startswith("## "):
                continue

            # 提取型号
            model_match = re.search(r"型号：(.+)", block)
            series_match = re.search(r"系列：(.+)", block)

            model = model_match.group(1).strip() if model_match else "未知型号"
            series = series_match.group(1).strip() if series_match else "未知系列"

            full_text = block.strip()
            if len(full_text) > 50:
                doc = Document(
                    page_content=full_text,
                    metadata={
                        "source": "kb_products.txt",
                        "type": "product",
                        "series": series,
                        "model": model,
                    },
                )
                documents.append(doc)
                logger.info(f"Loaded: product - {series} {model}")

    # 加载 pre_sales（按 ## FAQ 分割）
    presales_path = os.path.join(data_dir, "kb_pre_sales.txt")
    if os.path.exists(presales_path):
        with open(presales_path, "r", encoding="utf-8") as f:
            content = f.read()

        faq_blocks = re.split(r"(?=## FAQ\d+)", content)
        for block in faq_blocks:
            if not block.strip():
                continue
            full_text = block.strip()
            if len(full_text) > 20:
                # 提取 FAQ 编号作为 topic
                topic_match = re.search(r"## FAQ(\d+)", block)
                topic = "FAQ" + topic_match.group(1) if topic_match else "选购"
                doc = Document(
                    page_content=full_text,
                    metadata={
                        "source": "kb_pre_sales.txt",
                        "type": "pre_sales",
                        "topic": topic,
                    },
                )
                documents.append(doc)
                logger.info(f"Loaded: pre_sales {topic}")

    # 加载 after_sales（按 ## FAQ 分割）
    aftersales_path = os.path.join(data_dir, "kb_after_sales.txt")
    if os.path.exists(aftersales_path):
        with open(aftersales_path, "r", encoding="utf-8") as f:
            content = f.read()

        faq_blocks = re.split(r"(?=## FAQ\d+)", content)
        for block in faq_blocks:
            if not block.strip():
                continue
            full_text = block.strip()
            if len(full_text) > 20:
                topic_match = re.search(r"## FAQ(\d+)", block)
                topic = "FAQ" + topic_match.group(1) if topic_match else "售后"
                doc = Document(
                    page_content=full_text,
                    metadata={
                        "source": "kb_after_sales.txt",
                        "type": "after_sales",
                        "topic": topic,
                    },
                )
                documents.append(doc)
                logger.info(f"Loaded: after_sales {topic}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


class KnowledgeBase:
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
        return OllamaEmbeddings(
            model=config.embedding.model,
            base_url=config.embedding.base_url,
        )

    def initialize(self) -> None:
        if self._initialized:
            return

        os.makedirs(self.persist_directory, exist_ok=True)

        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            if self.vector_store._collection.count() > 0:
                logger.info(
                    f"Loaded existing knowledge base from {self.persist_directory}"
                )
            else:
                logger.warning(
                    "Knowledge base is empty. Run init_from_kb() to load data."
                )
        except Exception as e:
            logger.warning(f"Failed to load existing DB: {e}, creating new")
            self.vector_store = None

        self._initialized = True

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        self.initialize()
        return self.vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 3})

    def add_documents(self, documents: list[Document]) -> None:
        self.initialize()
        self.vector_store.add_documents(documents)

    def similarity_search(
        self, query: str, k: int = 3, filter: Optional[dict] = None
    ) -> list[Document]:
        self.initialize()
        return self.vector_store.similarity_search(query=query, k=k, filter=filter)


def get_rag() -> KnowledgeBase:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = KnowledgeBase()
    return _rag_instance


def init_from_files(
    data_dir: str = "./data", collection_name: str = "kefu_knowledge"
) -> int:
    """从 data/kb_*.txt 初始化向量数据库

    Args:
        data_dir: 数据文件目录
        collection_name: 向量库名称

    Returns:
        导入的文档数量
    """
    documents = load_kb_files(data_dir)

    if not documents:
        logger.warning(f"No documents loaded from {data_dir}")
        return 0

    persist_directory = config.chroma.persist_directory
    os.makedirs(persist_directory, exist_ok=True)

    embeddings = OllamaEmbeddings(
        model=config.embedding.model,
        base_url=config.embedding.base_url,
    )

    try:
        existing = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

        existing_hashes = set()
        existing_count = existing._collection.count()
        if existing_count > 0:
            all_docs = existing.get()
            for doc in all_docs.get("documents", []):
                existing_hashes.add(_compute_content_hash(doc))

        new_docs = []
        for doc in documents:
            doc_hash = _compute_content_hash(doc.page_content)
            if doc_hash not in existing_hashes:
                new_docs.append(doc)
                existing_hashes.add(doc_hash)

        if new_docs:
            existing.add_documents(new_docs)
            logger.info(f"Added {len(new_docs)} new documents")

    except Exception as e:
        logger.warning(f"Failed to load existing DB: {e}, creating new")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        new_docs = documents

    count = len(new_docs)
    logger.info(f"Vector DB update completed: {count} new documents")
    return count


if __name__ == "__main__":
    count = init_from_files("./data")
    print(f"Initialized {count} documents")
