# 智能客服 Agent

基于 LangChain + LangGraph 的 AI 智能客服聊天机器人，支持多轮对话、订单查询、物流查询、转接人工等功能。

## 功能特性

- **多轮对话**: 支持上下文记忆（Redis 存储）
- **RAG 知识库**: 基于 Chroma 向量库的智能问答
- **工具调用**: 查询订单、查询物流、转接人工
- **流式输出**: 打字机效果的真实流式响应
- **模型切换**: 支持本地 Ollama 和 SiliconFlow API
- **监控追踪**: LangSmith 支持

## 技术栈

- **LLM**: Ollama (qwen3.5:4b) / SiliconFlow (Qwen3.5-4B)
- **框架**: LangChain + LangGraph
- **向量库**: Chroma
- **嵌入**: Ollama (bge-m3)
- **前端**: Streamlit
- **存储**: Redis (对话记忆)

## 项目结构

```
kefu_agent/
├── app.py                      # Streamlit Web 入口
├── src/
│   ├── config/
│   │   ├── settings.py          # 配置加载 (dataclass)
│   │   └── logger.py            # 日志配置
│   ├── models/
│   │   └── types.py             # 数据类型定义
│   └── services/
│       ├── agent/               # Agent 核心
│       ├── llm/                 # LLM 管理
│       ├── tools/               # 工具定义
│       ├── memory/              # 对话记忆
│       └── rag/                 # RAG 知识库
├── tests/
│   └── test_services.py         # 单元测试
├── .env                         # 环境变量
├── .env.example                 # 环境变量模板
├── requirements.txt             # 依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件：

```env
# LLM 配置
LLM_PROVIDER=ollama
OLLAMA_API_KEY=ollama

# SiliconFlow (可选)
SILICONFLOW_API_KEY=your-api-key

# LangSmith 追踪 (可选)
LANGSMITH_API_KEY=your-langsmith-key

# Redis (可选)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. 启动 Ollama

```bash
# 确保模型已下载
ollama pull qwen3.5:4b
ollama pull bge-m3
```

### 4. 启动应用

```bash
streamlit run app.py
```

访问 http://localhost:8501

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `LLM_PROVIDER` | LLM 提供商 (ollama/siliconflow) | ollama |
| `LLM_MODEL` | 模型名称 | qwen3.5:4b |
| `OLLAMA_BASE_URL` | Ollama 地址 | http://localhost:11434 |
| `EMBEDDING_MODEL` | 嵌入模型 | bge-m3 |
| `LANGSMITH_API_KEY` | LangSmith API Key | - |
| `REDIS_HOST` | Redis 地址 | localhost |
| `REDIS_PORT` | Redis 端口 | 6379 |

## 测试

```bash
pytest tests/ -v
```

## 使用说明

- 输入商品相关问题进行咨询
- 输入订单号查询订单（如：订单号1234567890）
- 输入快递单号查询物流
- 输入"转人工"转接客服