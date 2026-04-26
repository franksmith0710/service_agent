"""
项目配置模块

从环境变量加载配置，支持多种 LLM 提供商（ollama / siliconflow）
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM 配置"""

    provider: str = "ollama"
    model: str = "deepseek-r1:1.5b"
    temperature: float = 0.7
    base_url: str = "http://localhost:11434"
    api_key: str = "ollama"


@dataclass
class SiliconFlowConfig:
    """SiliconFlow 配置"""

    base_url: str = "https://api.siliconflow.cn/v1"
    api_key: Optional[str] = None
    model: str = "Qwen/Qwen3.5-9B"


@dataclass
class LangSmithConfig:
    """LangSmith 配置"""

    api_key: Optional[str] = None
    project_name: str = "kefu-agent"





@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""

    model: str = "bge-m3"
    base_url: str = "http://localhost:11434"


@dataclass
class ChromaConfig:
    """Chroma 向量库配置"""

    persist_directory: str = "./data/chroma"


@dataclass
class PostgresConfig:
    """PostgreSQL 配置"""

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    database: str = "kefu_agent"


@dataclass
class ToolsConfig:
    """工具配置"""

    enabled: list[str] = field(
        default_factory=lambda: ["query_order", "query_logistics", "transfer_to_human"]
    )


@dataclass
class AppConfig:
    """应用全局配置"""

    llm_provider: str = "ollama"
    llm: LLMConfig = field(default_factory=LLMConfig)
    siliconflow: SiliconFlowConfig = field(default_factory=SiliconFlowConfig)
    langsmith: LangSmithConfig = field(default_factory=LangSmithConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)


def load_config() -> AppConfig:
    """从环境变量加载配置"""
    return AppConfig(
        llm_provider="siliconflow",
        llm=LLMConfig(
            provider="ollama",
            model="deepseek-r1:1.5b",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        ),
        siliconflow=SiliconFlowConfig(
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            model=os.getenv("LLM_MODEL", "Qwen/Qwen3.5-9B"),
        ),
        langsmith=LangSmithConfig(
            api_key=os.getenv("LANGCHAIN_API_KEY"),
            project_name=os.getenv("LANGCHAIN_PROJECT", "kefu-agent"),
        ),
        embedding=EmbeddingConfig(
            model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ),
        chroma=ChromaConfig(
            persist_directory=os.getenv("CHROMA_DIR", "./chroma_db"),
        ),
        postgres=PostgresConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database=os.getenv("POSTGRES_DB", "kefu_agent"),
        ),
        tools=ToolsConfig(
            enabled=[
                "query_order",
                "query_logistics",
                "transfer_to_human",
            ]
        ),
    )


config = load_config()
