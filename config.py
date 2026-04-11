import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

LLM_CONFIG = {
    "provider": LLM_PROVIDER,
    "model": "qwen3.5:4b",
    "temperature": 0.7,
    "base_url": "http://localhost:11434",
    "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
}

SILICONFLOW_CONFIG = {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": os.getenv("SILICONFLOW_API_KEY"),
    "model": "Qwen/Qwen2.5-4B-Instruct",
}

LANGSMITH_CONFIG = {
    "api_key": os.getenv("LANGSMITH_API_KEY", ""),
    "project_name": "kefu-agent",
}

EMBEDDING_CONFIG = {
    "model": "bge-m3",
    "base_url": "http://localhost:11434",
}

CHROMA_CONFIG = {
    "persist_directory": "./data/chroma",
}

TOOLS_CONFIG = {
    "enabled": [
        "query_order",
        "query_logistics",
        "transfer_to_human",
    ]
}
