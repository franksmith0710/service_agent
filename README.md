# 智能客服 Agent

基于 LangChain + LangGraph 的 AI 智能客服聊天机器人，支持多轮对话、订单查询、物流查询、转接人工等功能。

---

## 功能特性

- **多轮对话**: 基于内存存储的对话历史，跨轮次保持上下文
- **双 LLM 架构**:
  - 调度模型 (deepseek-r1:1.5b): 意图识别、路由决策、槽位提取
  - 生成模型 (Qwen3.5): 整合信息生成自然回复
- **槽位管理**: 自动提取并累积手机号、订单号、产品型号、故障类型
- **指代消解**: 智能解析"那款"、"它"、"刚才那个"等指代词
- **规则优先调度**: 明确场景（订单查询、转人工等）直接走规则，不调用模型
- **RAG 知识库**: Chroma 向量库 + BM25 多路召回，支持品牌、产品、售前、售后知识
- **工具调用**:
  - query_order: 订单查询（支持订单号/手机号/模糊搜索）
  - query_logistics: 物流查询（支持订单号/手机号）
  - query_user_info: 用户信息查询
  - transfer_to_human: 转接人工客服
- **流式输出**: 字符级流式响应
- **会话控制**:
  - 8轮上限自动转人工
  - 用户中断支持（"算了"、"换个话题"等）
  - 轮次自动计数
- **模型切换**: 支持本地 Ollama 和 SiliconFlow API
- **监控追踪**: LangSmith 支持（可选）

---

## 技术栈

- **LLM**: Ollama (deepseek-r1:1.5b) / SiliconFlow (Qwen/Qwen3.5-9B)
- **框架**: LangChain + LangGraph (StateGraph)
- **向量库**: Chroma
- **嵌入模型**: Ollama (bge-m3)
- **数据库**: PostgreSQL (结构化数据) + Chroma (向量数据)
- **前端**: Streamlit / FastAPI
- **存储**: 内存存储 (对话记忆)

---

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
│   │   └── types.py            # AgentState 定义 + Reducer
│   └── services/
│       ├── agent.py            # Agent 核心 (Graph + 入口)
│       ├── llm.py              # LLM 管理 (单例模式)
│       ├── tools.py            # 工具定义 (@tool 装饰器)
│       ├── memory.py           # 对话记忆 (内存存储)
│       ├── rag.py              # RAG 知识库 (多路召回)
│       ├── intent.py           # 意图识别 + 槽位提取 + 指代消解
│       └── postgres.py         # PostgreSQL 连接池
├── data/                       # 知识库数据
│   ├── kb_brand.txt            # 品牌信息
│   ├── kb_products.txt         # 产品信息
│   ├── kb_pre_sales.txt        # 售前 FAQ
│   └── kb_after_sales.txt      # 售后 FAQ
├── chroma_db/                  # Chroma 向量库持久化目录
├── tests/                      # 单元测试
└── README.md
```

---

## Agent 架构

### 双 LLM 设计

1. **调度模型 (llm_dispatch)**
   - 模型: deepseek-r1:1.5b (默认)
   - 职责: 意图识别、路由决策、工具调用判断
   - 输出: JSON 格式 (need_rag, need_tool, need_clarify, tool_calls)

2. **生成模型 (llm_with_tools)**
   - 模型: Ollama (qwen3.5) 或 SiliconFlow (Qwen/Qwen3.5-9B)
   - 职责: 整合 RAG 结果 + 工具结果 + 历史记录，生成最终回复
   - 特点: 绑定工具，可流式输出

### 完整消息流程

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. 加载记忆                                             │
│    - slots (手机号、订单号、产品、故障类型)              │
│    - context_entity (消解后的实体)                      │
│    - history (对话历史)                                 │
│    - turn_count (轮次计数)                              │
│    - session_status (会话状态)                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. 轮次检查                                             │
│    - turn_count >= 8 → 转人工 + 清除记忆 → END          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. 中断检查                                             │
│    - 检测 "算了"、"换个话题"、"先不提了" 等             │
│    - 是 → 保存状态 → "好的，保存进度。" → END           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 4. 指代消解                                             │
│    - 解析 "那款"、"它"、"刚才那个订单" 等               │
│    - 根据 context_entity 替换为具体实体                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 5. llm_dispatch 决策 (规则优先 + 模型兜底)              │
│    ─────────────────────────────────────────            │
│    规则覆盖场景:                                         │
│    - 有订单号/手机号 → 自动添加查询工具                   │
│    - 提到订单/物流但缺信息 → need_clarify                │
│    - 产品相关关键词 → need_rag                           │
│    - 转人工关键词 → transfer_to_human                   │
│    - 问候/感谢/再见 → 固定回复                           │
│    ─────────────────────────────────────────            │
│    兜底: 规则无法覆盖 → 调用小模型                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 6. 分支处理                                             │
│    ─────────────────────────────────────────            │
│    chat (闲聊) → 固定回复 → 保存 → END                  │
│    need_clarify → 追问 → 保存 → END                     │
│    need_tool=true → 进入 Graph 执行工具                  │
│    need_rag=true → 进入 Graph 执行 RAG                   │
│    need_tool + need_rag → 都执行                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 7. Graph 执行 (LangGraph StateGraph)                    │
│    ─────────────────────────────────────────            │
│    check_slots: 校验槽位完整性                           │
│       ↓                                                 │
│    (需要追问?) → clarify → END                          │
│       ↓                                                 │
│    tools: 循环执行工具 (最多4个)                         │
│       ├─ query_order                                    │
│       ├─ query_logistics                                │
│       ├─ query_user_info                                │
│       └─ transfer_to_human                              │
│       ↓                                                 │
│    (需要 RAG?) → rag: 多路召回检索                      │
│       ├─ 向量召回 (vector)                              │
│       └─ BM25 关键词召回                                 │
│       ↓                                                 │
│    summary: 汇总处理                                    │
│       ├─ 脱敏 (手机号、订单号)                          │
│       ├─ 去重 (同名工具只保留最后结果)                   │
│       └─ 截断 (RAG 结果封顶 2 条)                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 8. 生成模型生成回复                                      │
│    - SystemMessage: 你是机械革命官方客服...              │
│    - History: 对话历史                                   │
│    - RAG 结果: 参考资料                                  │
│    - 工具结果: 工具查询结果                              │
│    - UserMessage: 用户输入                               │
│    → 流式输出最终回复                                    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 9. 保存记忆                                             │
│    - slots (累加合并)                                   │
│    - context_entity                                     │
│    - session_status                                     │
│    - intent                                             │
│    - turn_count++                                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
   END
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件:

```env
# LLM 配置
LLM_PROVIDER=ollama
LLM_MODEL=deepseek-r1:1.5b
OLLAMA_BASE_URL=http://localhost:11434

# SiliconFlow 配置 (可选)
SILICONFLOW_API_KEY=your-api-key
SILICONFLOW_MODEL=Qwen/Qwen3.5-9B

# 嵌入模型
EMBEDDING_MODEL=bge-m3

# Chroma 向量库
CHROMA_DIR=./chroma_db

# PostgreSQL 配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
POSTGRES_DB=kefu_agent

# LangSmith 追踪 (可选)
LANGCHAIN_API_KEY=your-api-key
LANGCHAIN_PROJECT=kefu-agent
```

### 3. 初始化知识库

```bash
python -m src.services.rag
```

或启动应用时自动检测并初始化。

### 4. 启动服务

```bash
# Streamlit Web 界面
streamlit run app.py

# 或 FastAPI 接口
uvicorn api:app --reload
```

---

## API 接口

### 1. 根路径

```bash
GET /
```

返回:
```json
{
  "message": "智能客服 API",
  "version": "1.0.0",
  "llm_provider": "ollama"
}
```

### 2. 健康检查

```bash
GET /health
```

返回:
```json
{
  "status": "healthy"
}
```

### 3. 对话接口

```bash
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

### 4. 清除记忆

```bash
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

### 5. 获取对话历史

```bash
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

---

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| LLM_PROVIDER | LLM 提供商，可选 ollama/siliconflow | ollama |
| LLM_MODEL | 调度模型名称 | deepseek-r1:1.5b |
| OLLAMA_BASE_URL | Ollama 服务地址 | http://localhost:11434 |
| SILICONFLOW_API_KEY | SiliconFlow API Key | - |
| SILICONFLOW_MODEL | SiliconFlow 生成模型 | Qwen/Qwen3.5-9B |
| EMBEDDING_MODEL | 嵌入模型名称 | bge-m3 |
| CHROMA_DIR | Chroma 持久化目录 | ./chroma_db |
| POSTGRES_HOST | PostgreSQL 地址 | localhost |
| POSTGRES_PORT | PostgreSQL 端口 | 5432 |
| POSTGRES_USER | PostgreSQL 用户 | postgres |
| POSTGRES_PASSWORD | PostgreSQL 密码 | postgres |
| POSTGRES_DB | PostgreSQL 数据库名 | kefu_agent |
| LANGCHAIN_API_KEY | LangSmith API Key (可选) | - |
| LANGCHAIN_PROJECT | LangSmith 项目名称 | kefu-agent |

### 配置类 (settings.py)

```python
@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "deepseek-r1:1.5b"
    temperature: float = 0.7
    base_url: str = "http://localhost:11434"

@dataclass
class SiliconFlowConfig:
    base_url: str = "https://api.siliconflow.cn/v1"
    api_key: Optional[str] = None
    model: str = "Qwen/Qwen3.5-9B"

@dataclass
class ChromaConfig:
    persist_directory: str = "./chroma_db"

@dataclass
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    database: str = "kefu_agent"
```

---

## 数据库结构

### 表: users (用户信息)

| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | VARCHAR | 用户ID (主键) |
| username | VARCHAR | 用户名 |
| phone | VARCHAR | 手机号 |
| membership | VARCHAR | 会员等级 |
| points | INTEGER | 积分 |

### 表: orders (订单信息)

| 字段 | 类型 | 说明 |
|------|------|------|
| order_id | VARCHAR | 订单号 (主键) |
| user_id | VARCHAR | 用户ID |
| status | VARCHAR | 订单状态 |
| item_name | VARCHAR | 商品名称 |
| quantity | INTEGER | 数量 |
| price | DECIMAL | 价格 |
| total_amount | DECIMAL | 总金额 |
| pay_method | VARCHAR | 支付方式 |
| shipping_address | VARCHAR | 收货地址 |
| created_at | TIMESTAMP | 创建时间 |

### 表: logistics (物流信息)

| 字段 | 类型 | 说明 |
|------|------|------|
| order_id | VARCHAR | 订单号 (主键) |
| carrier | VARCHAR | 承运商 |
| tracking_number | VARCHAR | 运单号 |
| status | VARCHAR | 物流状态 |
| current_location | VARCHAR | 当前地点 |
| trace | JSON | 物流轨迹 |

---

## 工具定义 (tools.py)

### @tool query_order

```python
@tool
def query_order(
    order_id: str = "",      # 订单号 (优先)
    user_id: Optional[str] = None,
    phone: Optional[str] = None
) -> str
```

返回格式:
```
订单号: 1234567890
状态: 已发货
商品:
  - 机械革命 蛟龙16 Pro x1 ¥7999
总金额: ¥7999
下单时间: 2024-01-15 10:30:00
支付方式: 支付宝
收货地址: 北京市朝阳区xxx
```

### @tool query_logistics

```python
@tool
def query_logistics(
    order_id: str = "",
    phone: Optional[str] = None
) -> str
```

返回格式:
```
订单号: 1234567890
承运商: 顺丰速运
运单号: SF1234567890
当前状态: 配送中

物流轨迹:
  [2024-01-16 14:30] 北京市朝阳区 - 已签收
  [2024-01-16 09:20] 北京市朝阳区 - 配送中
  [2024-01-15 20:00] 北京市海淀区 - 已发出
```

### @tool query_user_info

```python
@tool
def query_user_info(phone: str) -> str
```

返回格式:
```
用户ID: U123456
用户名: 张三
手机号: 138****1234
会员等级: 金卡会员
积分: 5000
```

### @tool transfer_to_human

```python
@tool
def transfer_to_human(
    reason: str,
    conversation_summary: Optional[str] = None
) -> str
```

返回格式:
```
已为您创建转接工单，工单号: TK20240115103000
转接原因: 用户主动要求转人工
请稍候，人工客服将尽快为您服务。
人工客服工作时间: 周一至周五 9:00-18:00
客服热线: 400-990-5898
```

---

## 槽位提取 (intent.py)

### 正则提取

```python
# 手机号
r"1[3-9]\d{9}"

# 订单号
r"\d{10,}"
```

### 关键词匹配

产品型号:
- 耀世、蛟龙、极光、无界、旷世、翼龙、深海、泰坦

故障类型:
- 蓝屏、死机、黑屏、无法开机、充电、电池、发热、风扇、花屏、闪退、卡顿、重装、驱动

### 指代消解

支持的指代词:
- 那款、这款、它、刚才那个、刚才那个订单、上一个、上一个订单

消解规则:
- 根据 context_entity 中的 last_product、last_order、last_phone 进行替换

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

所有 Graph 节点使用 @safe_node 装饰器，异常时返回空字典，不中断流程。

---

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_services.py -v
```

---

## 注意事项

1. 内存存储: 对话记忆存在进程内存中，重启服务会丢失
2. 轮次限制: 8轮后自动转人工，清除记忆
3. 槽位校验: 查询订单/物流必须有 order_id 或 phone
4. 脱敏处理: 手机号显示为 1**********，订单号显示为 **********
5. 规则优先: 明确场景不走模型，减少 token 消耗