# 智能客服 Agent

基于 LangChain + LangGraph 的 AI 智能客服系统，支持多轮对话、订单查询、物流查询、转接人工等功能。

---

## 功能特性

- **多轮对话**: 基于内存存储的对话历史，跨轮次保持上下文
- **双 LLM 架构**:
  - 调度模型: 云端推理模型，负责意图识别、路由决策、槽位提取、工具调用
  - 生成模型: 云端生成模型，负责整合信息生成自然回复
- **槽位管理**: 自动提取并累积手机号、订单号、产品型号、故障类型
- **指代消解**: 智能解析"那款"、"它"、"刚才那个"等指代词
- **规则优先调度**: 明确场景（订单查询、转人工等）直接走规则，不调用模型
- **RAG 知识库**: Chroma 向量库 + BM25 多路召回，支持品牌、产品、售前、售后知识
- **工具调用**:
  - `query_order`: 订单查询（支持订单号/手机号/模糊搜索）
  - `query_logistics`: 物流查询（支持订单号/手机号）
  - `query_user_info`: 用户信息查询
  - `transfer_to_human`: 转接人工客服
- **流式输出**: token 级流式响应
- **会话控制**:
  - 10轮上限自动转人工
  - 用户中断支持（"算了"、"换个话题"等）
  - 轮次自动计数
- **工单管理**: 转人工后生成工单，支持查询和处理

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 调度 LLM | 云端推理模型 | 快速决策，负责意图识别与工具调用 |
| 生成 LLM | 云端生成模型 | 高质量文本生成，整合多源信息回复 |
| 框架 | LangChain + LangGraph (StateGraph) | Agent 流程编排 |
| 向量库 | Chroma | 本地向量存储 |
| 嵌入模型 | Ollama (bge-m3) | 本地嵌入 |
| 数据库 | PostgreSQL | 结构化数据存储 |
| 接口 | FastAPI | RESTful API |
| 前端 | Streamlit | Web 聊天界面 |
| 部署 | Docker + docker-compose | 容器化部署 |

---

## 项目结构

```
kefu_agent/
├── app.py                      # Streamlit Web 入口
├── api.py                      # FastAPI 接口
├── Dockerfile                  # Docker 镜像构建
├── docker-compose.yml          # 多容器编排
├── .env                        # 环境变量
├── .env.example                # 环境变量示例
├── .gitignore                  # Git 忽略配置
├── .dockerignore               # Docker 忽略配置
├── requirements.txt            # Python 依赖
├── README.md                   # 项目文档
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py         # 配置加载 (dataclass)
│   │   └── logger.py           # 日志配置
│   ├── models/
│   │   ├── __init__.py
│   │   └── types.py            # AgentState + DispatchResult 定义
│   └── services/
│       ├── __init__.py
│       ├── agent.py            # Agent 核心 (Graph + 入口)
│       ├── llm.py             # LLM 管理 (单例模式，双实例)
│       ├── tools.py           # 工具定义 (@tool 装饰器)
│       ├── memory.py           # 对话记忆 (内存存储)
│       ├── rag.py             # RAG 知识库 (多路召回)
│       ├── intent.py           # 意图识别 + 槽位提取 + 指代消解
│       ├── prompts.py          # Prompt 与关键词配置
│       └── postgres.py        # PostgreSQL 连接池 + 数据查询
├── data/                      # 数据与知识库
│   ├── chroma/               # Chroma 向量库持久化
│   ├── ollama/                # Ollama 模型存储
│   ├── postgres/              # PostgreSQL 数据持久化
│   ├── kb_brand.txt          # 品牌信息
│   ├── kb_products.txt       # 产品信息
│   ├── kb_pre_sales.txt      # 售前 FAQ
│   └── kb_after_sales.txt    # 售后 FAQ
├── docs/                      # 文档
│   └── init_db.sql.example   # 数据库初始化脚本示例
├── tests/                    # 单元测试
│   ├── __init__.py
│   └── test_services.py
└── scripts/                  # 脚本工具
    ├── docker-entrypoint.sh  # Docker 启动脚本
    ├── init_all.py          # 一键初始化
    └── insert_users.py       # 测试数据插入
```

---

## Agent 架构

### 双 LLM 设计

1. **调度模型 (`get_llm()`)**
   - 模型: 云端推理模型
   - 职责: 意图识别、路由决策、工具调用判断
   - 特点: 绑定工具，可触发 ReAct 执行
   - 输出: JSON 格式 (`need_rag`, `need_tool`, `need_clarify`, `tool_calls`)

2. **生成模型 (`get_llm_for_generation()`)**
   - 模型: 云端生成模型
   - 职责: 整合 RAG 结果 + 工具结果 + 历史记录，生成最终回复
   - 特点: 不绑定工具，纯文字生成，可流式输出
   - 优势: 避免工具误触发，保证生成质量

### 为什么分离两个模型？

- **绑定工具的模型会误触发**: 当生成模型看到手机号、订单号等数字时，可能误认为要调用工具
- **调度需要快速响应**: 调度模型只需做决策，不需要生成高质量文本
- **生成需要高输出质量**: 生成模型需要长文本自然回复，不应被工具定义干扰

### 完整消息流程

```
用户输入
  ├─ 加载记忆 (slots, context_entity, history, turn_count, session_status)
  ├─ 轮次检查 (turn_count >= 10→ 转人工)
  ├─ 中断检测 ("算了"、"换个话题"等 → 保存状态)
  ├─ 指代消解 (解析"那款"、"它"等)
  ├─ llm_dispatch 决策 (规则优先 + 模型兜底)
  │   ├─ 规则覆盖: 订单/物流/产品/转人工/问候感谢
  │   └─ 兜底: 无法覆盖 → 小模型决策
  ├─ Graph 执行
  │   ├─ check_slots: 校验槽位
  │   ├─ tools: 执行工具 (最多4个)
  │   ├─ rag: 多路召回 (向量 + BM25)
  │   └─ summary: 汇总处理 (脱敏/去重)
  ├─ 生成回复 (整合 RAG + 工具结果)
  └─ 保存记忆 (累加 slots, turn_count++)
```

---

## 快速开始

### 方式一：本地运行

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 配置环境变量

复制示例文件并填写实际值：

```bash
cp .env.example .env
# 然后编辑 .env 填写 API Key 等配置
```

主要配置项：

```env
# LLM API Key（必需）
SILICONFLOW_API_KEY=your_api_key_here

# 生成模型
LLM_MODEL=云端生成模型

# 调度模型
DISPATCH_MODEL=云端推理模型

# PostgreSQL 配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=kefu_agent

# Ollama 嵌入服务
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3
```

#### 3. 启动服务

```bash
# 启动 Streamlit Web 界面（端口 8501）
streamlit run app.py

# 或启动 FastAPI 接口（端口 8000）
uvicorn api:app --reload

# 或同时启动（需要两个终端）
```

### 方式二：Docker 部署（推荐）

#### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，确保 POSTGRES_PASSWORD 与 docker-compose.yml 中一致
```

#### 2. 构建并启动容器

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看日志
docker-compose logs -f app

# 停止服务
docker-compose down
```

服务端口：
- FastAPI: http://localhost:9090
- Streamlit: http://localhost:9091
- PostgreSQL: localhost:5432
- Ollama: http://localhost:11435

#### 3. 初始化数据（首次运行）

```bash
# 进入应用容器
docker-compose exec app bash

# 初始化知识库
python -m src.services.rag

# 插入测试数据（可选）
python scripts/insert_users.py
```

---

## API 接口

### 基础信息

- **基础路径**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs (Swagger UI，自动生成)
- **版本**: 1.0.0

### 接口列表

#### 1. 根路径

```http
GET /
```

返回:
```json
{
  "message": "智能客服 API",
  "version": "1.0.0",
  "dispatch_model": "云端推理模型",
  "generation_provider": "云端",
  "generation_model": "云端生成模型"
}
```

#### 2. 健康检查

```http
GET /health
```

返回:
```json
{
  "status": "healthy"
}
```

#### 3. 对话接口

```http
POST /chat
Content-Type: application/json

{
  "session_id": "abc123",
  "message": "查一下我的订单"
}
```

返回:
```json
{
  "session_id": "abc123",
  "response": "请问您的订单号是多少？或提供手机号也可以查询。",
  "success": true,
  "error": null
}
```

#### 4. 清除对话记忆

```http
POST /clear
Content-Type: application/json

{
  "session_id": "abc123"
}
```

返回:
```json
{
  "success": true,
  "message": "Session abc123 cleared"
}
```

#### 5. 获取对话历史

```http
GET /history/{session_id}
```

返回:
```json
{
  "session_id": "abc123",
  "history": [
    {"type": "user", "content": "查一下我的订单"},
    {"type": "assistant", "response": "请问您的订单号是多少？"}
  ]
}
```

#### 6. 工单管理接口

```http
# 获取工单列表
GET /tickets?status=pending&limit=50

# 获取单个工单
GET /tickets/{ticket_id}

# 标记工单为已处理
POST /tickets/{ticket_id}/resolve
```

---

## 配置说明

### 环境变量详解

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_PROVIDER` | LLM 提供商 | `siliconflow` |
| `LLM_MODEL` | 生成模型名称 | `云端生成模型` |
| `DISPATCH_MODEL` | 调度模型名称 | `云端推理模型` |
| `SILICONFLOW_API_KEY` | LLM API Key (必需) | - |
| `SILICONFLOW_BASE_URL` | LLM API 地址 | `https://api.siliconflow.cn/v1` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `EMBEDDING_MODEL` | 嵌入模型名称 | `bge-m3` |
| `CHROMA_DIR` | Chroma 持久化目录 | `./chroma_db` |
| `POSTGRES_HOST` | PostgreSQL 地址 | `localhost` |
| `POSTGRES_PORT` | PostgreSQL 端口 | `5432` |
| `POSTGRES_USER` | PostgreSQL 用户 | `postgres` |
| `POSTGRES_PASSWORD` | PostgreSQL 密码 | `postgres` |
| `POSTGRES_DB` | PostgreSQL 数据库名 | `kefu_agent` |
| `LANGCHAIN_TRACING_V2` | 启用 LangSmith 追踪 | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API Key (可选) | - |
| `LANGCHAIN_PROJECT` | LangSmith 项目名称 | `kefu-agent` |

### 配置优先级

`src/config/settings.py` 中的默认值会被环境变量覆盖：

- 生成模型 provider: 当前固定为云端 LLM（不可配）
- Chroma 目录: 默认 `./data/chroma`，环境变量可覆盖
- 工具启用: 默认 `query_order, query_logistics, transfer_to_human`

---

## 数据库结构

### 表: users (用户信息)

| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | VARCHAR(50) | 用户ID (主键) |
| username | VARCHAR(100) | 用户名 |
| phone | VARCHAR(20) | 手机号 (唯一) |
| membership | VARCHAR(50) | 会员等级 |
| points | INTEGER | 积分 |
| created_at | TIMESTAMP | 创建时间 |

### 表: orders (订单信息)

> **注意**: 支持同一订单号包含多个商品，因此使用自增 `id` 作为主键

| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 自增主键 |
| order_id | VARCHAR(50) | 订单号 |
| user_id | VARCHAR(50) | 用户ID (外键) |
| status | VARCHAR(20) | 订单状态 |
| item_name | VARCHAR(200) | 商品名称 |
| quantity | INTEGER | 数量 |
| price | DECIMAL(10,2) | 单价 |
| total_amount | DECIMAL(10,2) | 总金额 |
| pay_method | VARCHAR(20) | 支付方式 |
| shipping_address | TEXT | 收货地址 |
| created_at | TIMESTAMP | 创建时间 |

### 表: logistics (物流信息)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | SERIAL | 自增主键 |
| order_id | VARCHAR(50) | 订单号 (唯一，外键) |
| carrier | VARCHAR(50) | 承运商 |
| tracking_number | VARCHAR(50) | 运单号 |
| status | VARCHAR(20) | 物流状态 |
| current_location | TEXT | 当前地点 |
| trace | JSONB | 物流轨迹 |

### 表: transfer_tickets (转人工工单)

| 字段 | 类型 | 说明 |
|------|------|------|
| ticket_id | VARCHAR(20) | 工单号 (主键) |
| user_id | VARCHAR(50) | 用户ID |
| phone | VARCHAR(20) | 手机号 |
| session_id | VARCHAR(50) | 会话ID |
| reason | VARCHAR(200) | 转人工原因 |
| summary | TEXT | 对话摘要 |
| status | VARCHAR(20) | 状态 (pending/processing/resolved) |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### 初始化数据库

数据库表结构参考 `docs/init_db.sql.example`。如果重新创建 PostgreSQL 容器，可手动执行：

```bash
psql -U postgres -d kefu_agent -f docs/init_db.sql.example
```

> **注意**: 本项目的 PostgreSQL 数据已持久化在 `./data/postgres` 中，通常无需重新初始化。

---

## 核心模块说明

### agent.py - Agent 核心

```python
def run_agent(session_id: str, user_input: str, enable_stream: bool = True):
    """
    运行 Agent 处理用户输入
    
    Args:
        session_id: 会话 ID
        user_input: 用户输入
        enable_stream: 是否启用流式输出
    
    Yields:
        流式输出的 token
    """
```

### tools.py - 工具定义

```python
@tool
def query_order(order_id: str = "", user_id: Optional[str] = None, phone: Optional[str] = None) -> str:
    """订单查询 (支持 order_id/user_id/phone/模糊搜索)"""

@tool
def query_logistics(order_id: str = "", phone: Optional[str] = None) -> str:
    """物流查询 (支持 order_id/phone)"""

@tool
def query_user_info(phone: str) -> str:
    """用户信息查询"""

@tool
def transfer_to_human(reason: str, conversation_summary: Optional[str] = None) -> str:
    """转接人工客服"""
```

### intent.py - 意图识别

```python
def llm_dispatch(state: AgentState) -> DispatchResult:
    """
    调度决策：规则优先 + 小模型兜底
    
    规则覆盖:
    - 有订单号/手机号 → 自动工具调用
    - 提到订单/物流但缺信息 → 追问
    - 产品相关 → RAG
    - 转人工 → 工具调用
    - 问候/感谢 → 固定回复
    
    兜底: 规则无法覆盖 → get_llm() 小模型决策
    """
```

### llm.py - LLM 管理

```python
def get_llm() -> BaseChatModel:
    """调度模型，绑定工具用于意图识别"""

def get_llm_for_generation() -> BaseChatModel:
    """生成模型，纯文字生成"""
```

---

## 槽位提取

### 正则提取

```python
# 手机号
r"1[3-9]\d{9}"

# 订单号
r"\d{10,}"
```

### 关键词匹配

**产品型号**: 耀世、蛟龙、极光、无界、旷世、翼龙、深海、泰坦、小白

**故障类型**: 蓝屏、死机、黑屏、无法开机、充电、电池、发热、风扇、花屏、闪退、卡顿、重装、驱动

### 指代消解

**支持的指代词**: 那款、这款、这个、那个、它、刚才那个、刚才那个订单、刚才那个产品、上一个、上一个订单

**消解规则**: 根据 `context_entity` 中的 `last_product`、`last_order`、`last_phone` 进行替换

---

## 错误处理

### 工具错误策略

```python
TOOL_ERROR_STRATEGY = {
    "query_order": "抱歉，订单查询服务暂时不可用，请稍后重试或拨打客服热线 400-990-5898",
    "query_logistics": "抱歉，物流查询服务暂时不可用，请稍后重试",
    "query_user_info": "抱歉，用户信息查询服务暂时不可用",
    "transfer_to_human": "抱歉，转接服务暂时不可用，请拨打客服热线 400-990-5898",
}
```

### Graph 节点异常

所有 Graph 节点使用 `@safe_node` 装饰器，异常时返回空字典，不中断流程。

---


## 注意事项

1. **内存存储**: 对话记忆存在进程内存中，重启服务会丢失（已知限制）
2. **轮次限制**: 10轮后自动转人工，清除记忆
3. **槽位校验**: 查询订单/物流必须有 order_id 或 phone
4. **脱敏处理**: 手机号显示为 `1**********`，订单号部分脱敏
5. **规则优先**: 明确场景不走模型，减少 token 消耗
6. **双 LLM 架构**: 调度模型绑定工具，生成模型不绑定，避免误触发
7. **数据持久化**: PostgreSQL 数据在 `./data/postgres`，Chroma 数据在 `./data/chroma`
8. **敏感信息**: `.env` 文件包含 API Key，已加入 `.gitignore`，请勿提交

---

## 常见问题

### Q: 为什么生成模型输出慢/token高？

可能原因：
1. 云端生成模型 API 响应较慢
2. 上下文太长（历史消息 + RAG 结果 + 工具结果）
3. 生成模型 max_tokens=512 限制，可根据需求调整

### Q: 如何切换生成模型？

修改 `.env` 中的 `LLM_MODEL` 配置为支持的模型名称（需与提供商对应）

### Q: 如何添加新工具？

1. 在 `tools.py` 中定义新工具（使用 `@tool` 装饰器）
2. 在 `get_all_tools()` 中添加
3. 在 `TOOL_MAP` 中注册（如果需要在 Graph 中执行）

### Q: 如何禁用某些工具？

修改 `settings.py` 中 `tools.enabled` 列表：

```python
tools=ToolsConfig(
    enabled=["query_order", "query_logistics", "transfer_to_human"]
)
```

### Q: Docker 部署后如何查看日志？

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f app
docker-compose logs -f db
docker-compose logs -f ollama
```

### Q: 如何清理 Docker 环境？

```bash
# 停止并删除容器、网络（保留卷）
docker-compose down

# 停止并删除容器、网络、卷（完全清理）
docker-compose down -v

# 重新构建镜像
docker-compose up -d --build
```

---

