"""
RAG 知识库服务模块

提供基于 Chroma 的向量知识库检索
支持从文件加载知识库数据（增量添加）
"""

import os
import glob
import logging
import hashlib
from typing import Optional, List, Dict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import config
from src.config.logger import get_logger

logger = get_logger(__name__)

_rag_instance: Optional["KnowledgeBase"] = None

CATEGORY_MAP = {
    "kb_brand.txt": "brand",
    "kb_products.txt": "products",
    "kb_pre_sales.txt": "pre_sales",
    "kb_after_sales.txt": "after_sales",
}


def _compute_content_hash(content: str) -> str:
    """计算内容哈希，用于去重"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


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
                # 如果为空，提示需要导入
                logger.warning(
                    "Knowledge base is empty. Run init_from_files() to load data."
                )
        except Exception as e:
            logger.warning(f"Failed to load existing DB: {e}, creating new")
            self.vector_store = None

        self._initialized = True  # 即使失败也标记为已初始化

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


def load_knowledge_files(data_dir: str = "./data") -> List[Document]:
    """从目录加载知识库文件

    Args:
        data_dir: 数据文件目录

    Returns:
        Document列表
    """
    documents = []

    for filename, category in CATEGORY_MAP.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if content.strip():
            doc = Document(
                page_content=content,
                metadata={"source": filename, "category": category},
            )
            documents.append(doc)
            logger.info(f"Loaded: {filename} ({category})")

    return documents


def init_from_files(
    data_dir: str = "./data", collection_name: str = "kefu_knowledge"
) -> int:
    """从文件初始化向量数据库（增量添加）

    Args:
        data_dir: 数据文件目录
        collection_name: 向量库名称

    Returns:
        导入的文档数量
    """
    documents = load_knowledge_files(data_dir)

    if not documents:
        logger.warning(f"No knowledge files found in {data_dir}")
        return 0

    persist_directory = config.chroma.persist_directory
    os.makedirs(persist_directory, exist_ok=True)

    embeddings = OllamaEmbeddings(
        model=config.embedding.model,
        base_url=config.embedding.base_url,
    )

    new_docs = []
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

        for doc in documents:
            doc_hash = _compute_content_hash(doc.page_content)
            if doc_hash not in existing_hashes:
                new_docs.append(doc)
                existing_hashes.add(doc_hash)

        if new_docs:
            existing.add_documents(new_docs)
            logger.info(
                f"Added {len(new_docs)} new documents (skipped {len(documents) - len(new_docs)} duplicates)"
            )

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
