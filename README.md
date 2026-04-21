# 智能客服 Agent

基于 LangChain + LangGraph 的 AI 智能客服聊天机器人，支持多轮对话、订单查询、物流查询、转接人工等功能。

## 功能特性

- **多轮对话**: 支持上下文记忆（内存存储）
- **槽位记忆**: 自动提取并累积手机号、订单号、产品型号
- **RAG 知识库**: 基于 Chroma 向量库的智能问答
- **LLM 决策**: llm_dispatch 判断是否需要 RAG/Tool/Clarify/Transfer
- **ReAct 工具调用**: 查询订单、查询物流、用户信息、转接人工
- **流式输出**: 字符级流式响应
- **轮次上限**: 8轮后自动转接人工
- **模型切换**: 支持本地 Ollama 和 SiliconFlow API
- **监控追踪**: LangSmith 支持

## 技术栈

- **LLM**: Ollama (qwen3.5:4b) / SiliconFlow (Qwen3.5-4B)
- **框架**: LangChain + LangGraph (StateGraph)
- **向量库**: Chroma
- **嵌入**: Ollama (bge-m3)
- **数据库**: PostgreSQL (结构化数据) + Chroma (向量数据)
- **前端**: Streamlit / FastAPI
- **存储**: Redis (对话记忆)

## 项目结构

```
kefu_agent/
├── app.py                      # Streamlit Web 入口
├── api.py                      # FastAPI 接口
├── src/
│   ├── config/
│   │   ├── settings.py         # 配置加载
│   │   └── logger.py           # 日志配置
│   ├── models/
│   │   └── types.py            # 类型定义 + Reducer
│   └── services/
│       ├── agent.py           # Agent 核心 (Graph + 入口)
│       ├── llm.py              # LLM 管理
│       ├── tools.py            # 工具定义
│       ├── memory.py           # 对话记忆
│       ├── rag.py              # RAG 知识库
│       ├── intent.py           # 槽位提取 + 消解
│       └── postgres.py          # PostgreSQL 
├── data/                      # 知识库数据
├── scripts/                   # 初始化脚本
├── tests/                     # 单元测试
└── README.md
```

## Agent 架构

### 双 LLM 设计

| LLM | 职责 | 输入 | 输出 |
|-----|------|------|------|
| llm_dispatch | 决策路由 | history + slots + context | need_rag / need_tool / need_clarify / need_transfer |
| llm_with_tools | 生成回复 | history + RAG结果 + user_input | 最终回复文本 |

### 消息流程

```
用户输入
    ↓
加载记忆 (slots, context_entity, history, turn_count)
    ↓
llm_dispatch 决策
    ↓
┌──────────────────────────────────────────────────┐
│ need_clarify → 要求补充信息 → 保存 → END           │
│ need_transfer → 转接人工 → 保存 → END           │
│ chat → 固定回复 → 保存 → END                    │
│ need_rag + need_tool → Graph 执行                │
│     intent → rag → RAG 检索                     │
│     agent → ReAct 循环 → tools                  │
└──────────────────────────────────────────────────┘
    ↓
llm_with_tools 生成回复
    ↓
保存记忆 (slots, context_entity, session_status, turn)
    ↓
END
```

### Graph 节点

| 节点 | 说明 |
|------|------|
| intent_node | 状态传递 (intent → rag / agent) |
| rag_node | RAG 检索，返回 rag_docs |
| agent_node | ReAct 循环节点 |
| tools_node | 工具执行，返回 tool_results + slots |

### 消息存储

| 存储 | 内容 | 说明 |
|------|------|------|
| memory.slots | {"phone": "...", "order_id": "..."} | 槽位（累加合并） |
| memory.context_entity | {"order_id": "订单号", "phone": "手机号"} | 消解后的实体 |
| memory.session_status | idle / waiting_slot / transfering | 对话状态 |
| memory.turn_count | int | 对话轮次 |

### 工具

| 工具 | 参数 | 返回 |
|------|------|------|
| query_order | order_id / phone | 订单详情 |
| query_logistics | order_id / phone | 物流信息 |
| query_user_info | phone | 用户信息 |
| transfer_to_human | reason | 转接确认 |

### 槽位提取

| 槽位 | 提取方式 | 模式 |
|------|---------|------|
| phone | regex | 1[3-9]\d{9} |
| order_id | regex | 订单号(\d{10,}) |
| product | 关键词匹配 | 机械革命... |

### 轮次上限

- 最大轮次: 8 轮
- 超过上限: 提示转接 + 清除记忆 + 设置 transfering 状态

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`:

```env
LLM_PROVIDER=ollama
SILICONFLOW_API_KEY=your-key
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
POSTGRES_DB=kefu_agent
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. 初始化

```bash
python scripts/init_all.py
```

### 4. 启动

```bash
streamlit run app.py
# 或
uvicorn api:app --reload
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /chat | 对话接口 |
| POST | /clear | 清除记忆 |
| GET | /health | 健康检查 |

### 请求示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "user_input": "查一下订单"}'
```

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| LLM_PROVIDER | LLM 提供商 | ollama |
| LLM_MODEL | 模型名称 | qwen3.5:4b |
| OLLAMA_BASE_URL | Ollama 地址 | http://localhost:11434 |
| EMBEDDING_MODEL | 嵌入模型 | bge-m3 |
| POSTGRES_HOST | PostgreSQL 地址 | localhost |
| REDIS_HOST | Redis 地址 | localhost |

## 测试

```bash
pytest tests/ -v
```

## 数据库

| 表 | 数据量 |
|---|-------|
| products | 7 |
| users | 5 |
| orders | 5 |
| logistics | 2 |