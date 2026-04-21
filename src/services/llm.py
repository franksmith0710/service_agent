"""
LLM 服务模块

提供 LLM 初始化和模型切换功能
使用单例模式管理实例
"""

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
    _llm: Optional[BaseChatModel] = None
    _llm_gen: Optional[BaseChatModel] = None  # 生成专用
    _llm_with_tools: Optional[BaseChatModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ========== 调度专用：极速 JSON ==========
    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = self._create_llm_dispatch()
        return self._llm

    # ========== 生成回答专用：正常说话 ==========
    def get_llm_for_generation(self) -> BaseChatModel:
        if self._llm_gen is None:
            self._llm_gen = self._create_llm_generation()
        return self._llm_gen

    def get_llm_with_tools(self, tools: list = None) -> BaseChatModel:
        if self._llm_with_tools is None:
            if tools is None:
                from src.services.tools import get_all_tools
                tools = get_all_tools()
            self._llm_with_tools = self.get_llm_for_generation().bind_tools(tools)
        return self._llm_with_tools

    def reset(self):
        self._llm = None
        self._llm_gen = None
        self._llm_with_tools = None
        logger.info("LLM manager reset")

    # ------------------------------
    # 调度模型：极速、短JSON、不废话
    # ------------------------------
    def _create_llm_dispatch(self) -> BaseChatModel:
        provider = config.llm_provider
        if provider == "siliconflow":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.siliconflow.model,
                base_url=config.siliconflow.base_url,
                api_key=config.siliconflow.api_key,
                temperature=0.2,
                max_tokens=120,
                timeout=20,
                max_retries=1,
                extra_body={
                    "top_p": 0.3,
                }
            )
        else:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.llm.model,
                base_url=config.llm.base_url,
                temperature=0.2,
                num_predict=120,
                timeout=20,
            )

    # ------------------------------
    # 生成模型：正常回答、能写长文本、自然
    # ------------------------------
    def _create_llm_generation(self) -> BaseChatModel:
        provider = config.llm_provider
        if provider == "siliconflow":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.siliconflow.model,
                base_url=config.siliconflow.base_url,
                api_key=config.siliconflow.api_key,
                temperature=0.7,
                max_tokens=1024,
                timeout=30,
                max_retries=2,
            )
        else:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.llm.model,
                base_url=config.llm.base_url,
                temperature=0.7,
                num_predict=1024,
                timeout=30,
            )


_llm_manager = LLMManager()

def get_llm() -> BaseChatModel:
    return _llm_manager.get_llm()

def get_llm_for_generation() -> BaseChatModel:
    return _llm_manager.get_llm_for_generation()

def get_llm_with_tools() -> BaseChatModel:
    return _llm_manager.get_llm_with_tools()

def reset_llm():
    _llm_manager.reset()