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
    """LLM 管理器（单例）"""

    _instance: Optional["LLMManager"] = None
    _llm: Optional[BaseChatModel] = None
    _llm_with_tools: Optional[BaseChatModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_llm(self) -> BaseChatModel:
        """获取 LLM 实例"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def get_llm_with_tools(self, tools: list = None) -> BaseChatModel:
        """获取绑定工具的 LLM 实例"""
        if self._llm_with_tools is None:
            if tools is None:
                from src.services.tools import get_all_tools

                tools = get_all_tools()
            self._llm_with_tools = self.get_llm().bind_tools(tools)
        return self._llm_with_tools

    def reset(self):
        """重置实例（用于测试）"""
        self._llm = None
        self._llm_with_tools = None
        logger.info("LLM manager reset")

    def _create_llm(self) -> BaseChatModel:
        """创建 LLM 实例"""
        provider = config.llm_provider

        if provider == "siliconflow":
            from langchain_openai import ChatOpenAI

            api_key = config.siliconflow.api_key
            if not api_key:
                raise ValueError("SILICONFLOW_API_KEY is not set")

            llm = ChatOpenAI(
                model=config.siliconflow.model,
                base_url=config.siliconflow.base_url,
                api_key=api_key,
                temperature=config.llm.temperature,
                timeout=config.llm.timeout,
                max_retries=2,
            )
            logger.info(f"Using SiliconFlow model: {config.siliconflow.model}")
            return llm
        else:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=config.llm.model,
                base_url=config.llm.base_url,
                temperature=config.llm.temperature,
                num_gpu=1,
                timeout=config.llm.timeout,
            )
            logger.info(f"Using Ollama model: {config.llm.model}")
            return llm


_llm_manager = LLMManager()


def get_llm() -> BaseChatModel:
    """获取 LLM 实例"""
    return _llm_manager.get_llm()


def get_llm_with_tools():
    """获取绑定工具的 LLM 实例"""
    return _llm_manager.get_llm_with_tools()


def reset_llm():
    """重置 LLM 实例"""
    _llm_manager.reset()
