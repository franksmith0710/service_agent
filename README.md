# 智能客服 Agent

基于 LangChain + LangGraph 的 AI 智能客服聊天机器人，支持多轮对话、订单查询、物流查询、转接人工等功能。

## 功能特性

- **多轮对话**: 支持上下文记忆（Redis 存储）
- **RAG 知识库**: 基于 Chroma 向量库的智能问答
- **意图识别**: 规则匹配 + 路由决策
- **ReAct 工具调用**: 查询订单、查询物流、用户信息、转接人工
- **流式输出**: 节点级别的流式响应
- **模型切换**: 支持本地 Ollama 和 SiliconFlow API
- **监控追踪**: LangSmith 支持

## 技术栈

- **LLM**: Ollama (qwen3.5:4b) / SiliconFlow (Qwen3.5-4B)
- **框架**: LangChain + LangGraph (StateGraph)
- **向量库**: Chroma
- **嵌入**: Ollama (bge-m3)
- **数据库**: PostgreSQL (结构化数据) + Chroma (向量数据)
- **前端**: Streamlit
- **存储**: Redis (对话记忆)

## 项目结构

```
kefu_agent/
├── app.py                      # Streamlit Web 入口
├── api.py                      # FastAPI 接口
├── src/
│   ├── config/
│   │   ├── settings.py         # 配置加载 (dataclass)
│   │   └── logger.py           # 日志配置
│   ├── models/
│   │   └── types.py            # 数据类型定义
│   └── services/
│       ├── agent.py           # Agent 核心 (LangGraph)
│       ├── llm.py              # LLM 管理
│       ├── tools.py            # 工具定义
│       ├── memory.py           # 对话记忆
│       ├── rag.py              # RAG 知识库
│       ├── intent.py           # 意图识别
│       ├── postgres.py          # PostgreSQL 数据库层
│       └── validator.py       # 输入验证
├── data/                      # 知识库数据
│   ├── kb_brand.txt          # 品牌信息
│   ├── kb_products.txt       # 产品参数
│   ├── kb_pre_sales.txt     # 售前FAQ
│   └── kb_after_sales.txt    # 售后FAQ
├── scripts/                   # 初始化脚本
│   ├── insert_products.py    # 插入产品数据
│   └── insert_users.py      # 插入用户数据
├── tests/
│   └── test_services.py      # 单元测试
├── .env                      # 环境变量
├── .env.example             # 环境变量模板
├── requirements.txt          # 依赖
├── agent.md                 # Agent 设计文档
├── jxgm.md                 # 机械革命知识库
└── README.md
```

## Agent 流程图

```
用户输入 → 加载记忆(Redis) → intent_node → router

router 路由:
├─ chat (问候/感谢) → chat_node → END
├─ product/pre_sales/after_sales → rag_node → END
├─ order/logistics/user → agent_node(ReAct循环) → tools_node → END
└─ transfer → transfer_node → END

保存记忆 → END
```

### 意图路由

| 意图 | 关键词 | 路由 |
|------|--------|------|
| chat | 你好、谢谢、再见 | chat |
| transfer | 转人工、投诉 | transfer |
| product | 产品、电脑、价格 | rag |
| order | 订单号、查订单 | agent |
| logistics | 物流、快递 | agent |
| user | 会员、积分 | agent |

### 工具

- query_order: 查询订单
- query_logistics: 查询物流
- query_user_info: 查询用户
- transfer_to_human: 转人工

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并编辑：

```env
# LLM 配置
LLM_PROVIDER=ollama

# SiliconFlow (可选)
SILICONFLOW_API_KEY=your-api-key

# LangSmith (可选)
LANGCHAIN_API_KEY=your-langsmith-key

# PostgreSQL 数据库
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
POSTGRES_DB=kefu_agent

# Redis (可选)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. 启动 PostgreSQL

确保 PostgreSQL 已安装并运行，创建数据库：

```bash
# 使用 psql 或 pgAdmin 创建数据库 kefu_agent
# 编码选择 UTF8
```

### 4. 启动 Ollama

```bash
# 确保模型已下载
ollama pull qwen3.5:4b
ollama pull bge-m3
```

### 5. 初始化数据

```bash
# 初始化产品数据
python scripts/insert_products.py

# 初始化用户和订单数据
python scripts/insert_users.py
```

### 6. 启动应用

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
| `POSTGRES_HOST` | PostgreSQL 地址 | localhost |
| `POSTGRES_PORT` | PostgreSQL 端口 | 5432 |
| `LANGCHAIN_API_KEY` | LangSmith API Key | - |
| `REDIS_HOST` | Redis 地址 | localhost |
| `REDIS_PORT` | Redis 端口 | 6379 |

## 测试

```bash
pytest tests/ -v
```

## 数据库数据

| 表名 | 数据量 | 说明 |
|------|-------|------|
| products | 7 | 机械革命全系列笔记本 |
| users | 5 | 测试用户 |
| orders | 5 | 订单数据 |
| logistics | 2 | 物流信息 |

## 知识库数据

| 文件 | category | 说明 |
|------|----------|------|
| kb_brand.txt | brand | 品牌信息 |
| kb_products.txt | products | 产品参数 |
| kb_pre_sales.txt | pre_sales | 售前FAQ |
| kb_after_sales.txt | after_sales | 售后FAQ |

## 使用说明

- 输入商品相关问题进行咨询（走 RAG）
- 输入订单号查询订单（如：订单号20250413001）
- 输入快递单号查询物流
- 输入"转人工"转接客服