import os
from typing import Optional, List
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import EMBEDDING_CONFIG, CHROMA_CONFIG


class KnowledgeBase:
    def __init__(
        self,
        collection_name: str = "kefu_knowledge",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or CHROMA_CONFIG["persist_directory"]
        self.embeddings = self._create_embeddings()
        self.vector_store = None

    def _create_embeddings(self):
        return OllamaEmbeddings(
            model=EMBEDDING_CONFIG["model"], base_url=EMBEDDING_CONFIG["base_url"]
        )

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        if self.vector_store is None:
            self._load_vector_store()

        return self.vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 3})

    def _load_vector_store(self):
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                client=self._create_chroma_client(),
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        else:
            self.vector_store = Chroma.from_documents(
                documents=[],
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
            )

def _create_chroma_client(self):
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Document]):
        if self.vector_store is None:
            self._load_vector_store()
        self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        if self.vector_store is None:
            self._load_vector_store()
        return self.vector_store.similarity_search(query, k=k)


def create_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


def get_default_docs() -> list[Document]:
    return [
        Document(
            page_content="我们的营业时间是周一至周五 9:00-18:00，节假日除外。",
            metadata={"source": "营业时间"},
        ),
        Document(
            page_content="退换货政策：商品支持7天无理由退换货，质量问题我们承担运费。",
            metadata={"source": "退换货政策"},
        ),
        Document(
            page_content="支持支付宝、微信支付、银行卡等多种支付方式。",
            metadata={"source": "支付方式"},
        ),
        Document(
            page_content="会员等级分为普通会员、银卡会员、金卡会员和钻石会员，不同等级享受不同折扣。",
            metadata={"source": "会员制度"},
        ),
        Document(
            page_content="订单问题请联系客服，提供订单号以便查询。",
            metadata={"source": "订单帮助"},
        ),
    ]
