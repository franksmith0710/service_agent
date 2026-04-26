"""
LLM 服务模块

提供 LLM 初始化和模型切换功能
使用单例模式管理实例
"""

from typing import Optional
from langchain_core.language_models import BaseChatModel

from src.config.settings import config
from src.config.logger import get_logger

logger = get_logger(__name__)


class LLMManager:
    _instance: Optional["LLMManager"] = None
    _llm_dispatch: Optional[BaseChatModel] = None
    _llm_gen: Optional[BaseChatModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_llm(self) -> BaseChatModel:
        """调度模型 - 决策用，直接绑定工具"""
        if self._llm_dispatch is None:
            from src.services.tools import get_all_tools
            self._llm_dispatch = self._create_llm_dispatch().bind_tools(get_all_tools())
        return self._llm_dispatch

    def get_llm_for_generation(self) -> BaseChatModel:
        """生成模型 - 永远不带工具，只生成文字"""
        if self._llm_gen is None:
            self._llm_gen = self._create_llm_generation()
        return self._llm_gen

    def reset(self):
        self._llm_dispatch = None
        self._llm_gen = None
        logger.info("LLM manager reset")

    def _create_llm_dispatch(self) -> BaseChatModel:
        """调度模型：DeepSeek-R1 推理模型"""
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="deepseek-r1:1.5b",
            base_url=config.llm.base_url,
            temperature=0.1,
            num_predict=512,
        )

    def _create_llm_generation(self) -> BaseChatModel:
        """生成模型：正常回答、能写长文本、自然"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.siliconflow.model,
            base_url=config.siliconflow.base_url,
            api_key=config.siliconflow.api_key,
            temperature=0.7,
            max_tokens=1024,
            request_timeout=30,
            max_retries=2,
        )


_llm_manager = LLMManager()


def get_llm() -> BaseChatModel:
    """调度模型 - 用于意图识别和决策"""
    return _llm_manager.get_llm()


def get_llm_for_generation() -> BaseChatModel:
    """生成模型 - 永远不带工具，只生成文字回答"""
    return _llm_manager.get_llm_for_generation()


def reset_llm():
    _llm_manager.reset()