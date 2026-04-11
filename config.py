import os

LLM_CONFIG = {
    "model": "qwen3.5:4b",
    "temperature": 0.7,
    "base_url": "http://localhost:11434",
    "api_key": "ollama",
}

LANGSMITH_CONFIG = {
    "api_key": "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",
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
