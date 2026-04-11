"""
LLM 服务模块

提供 LLM 初始化和模型切换功能
"""

from typing import Optional
from langchain_core.language_models import BaseChatModel

from src.config.settings import config
from src.config.logger import get_logger

logger = get_logger(__name__)

_llm_instance: Optional[BaseChatModel] = None


def get_llm() -> BaseChatModel:
    """
    获取 LLM 实例，根据配置选择模型

    Returns:
        LLM 实例
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    provider = config.llm_provider

    if provider == "siliconflow":
        from langchain_openai import ChatOpenAI

        api_key = config.siliconflow.api_key
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY is not set")

        _llm_instance = ChatOpenAI(
            model=config.siliconflow.model,
            base_url=config.siliconflow.base_url,
            api_key=api_key,
            temperature=config.llm.temperature,
        )
        logger.info(f"Using SiliconFlow model: {config.siliconflow.model}")

    else:
        from langchain_ollama import ChatOllama

        _llm_instance = ChatOllama(
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
        )
        logger.info(f"Using Ollama model: {config.llm.model}")

    return _llm_instance


def reset_llm():
    """重置 LLM 实例（用于切换模型）"""
    global _llm_instance
    _llm_instance = None
    logger.info("LLM instance reset")
