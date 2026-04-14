# 智能客服 Agent 改进计划

## 项目概述

基于对代码的全面审查，本文档记录了发现的问题及改进方案。

---

## 一、问题清单

### 🚨 P0 - 必须修复（核心 Bug）

| 编号 | 严重程度 | 模块 | 问题描述 | 影响 |
|------|----------|------|----------|------|
| B001 | 致命 | rag.py | `init_from_files()` 每次删除整个 collection | 知识库数据丢失 |
| B002 | 致命 | agent.py | ToolMessage 后消息历史未正确传递 | 工具结果无法被理解 |
| B003 | 严重 | agent.py | 流式输出只是简单拼接 | 用户体验极差 |
| B004 | 严重 | rag.py | 没有自动初始化逻辑 | 首次启动需手动执行 |

### ⚠️ P1 - 重要改进

| 编号 | 严重程度 | 模块 | 问题描述 | 影响 |
|------|----------|------|----------|------|
| I001 | 高 | intent.py | 纯规则匹配，无法处理模糊问句 | 意图误判 |
| I002 | 高 | postgres.py | `get_order_by_id` 只支持单商品 | 多商品订单展示不完整 |
| I003 | 中 | memory.py | Redis 无连接池，高并发不稳 | 连接泄漏风险 |
| I004 | 中 | tools.py | 没有输入验证 | 可能注入恶意查询 |

### 📝 P2 - 优化建议

| 编号 | 模块 | 建议 |
|------|------|------|
| O001 | 整体 | 缺少健康检查 endpoint |
| O002 | 整体 | 缺少 metrics 监控 |
| O003 | agent.py | ReAct 循环次数限制，防止死循环 |
| O004 | rag.py | 支持按 category 筛选检索 |

---

## 二、改进计划

### Phase 1: 核心 Bug 修复 🚨

#### B001: 修复 RAG 初始化逻辑

**问题**: `rag.py:173-181` 每次调用 `init_from_files()` 都会删除现有 collection

**修复方案**:
```python
# 修改为增量添加，不删除现有数据
def init_from_files(data_dir: str = "./data") -> int:
    # ...加载文档...
    # 检查是否存在，不存在则创建，存在则增量添加
    # 对每个 chunk 检查是否已存在（通过内容哈希）
```

**修改文件**: `src/services/rag.py`

---

#### B002: 修复 Agent 消息传递

**问题**: `agent.py:137-155` ToolMessage 后调用 LLM 时消息构建错误

**修复方案**:
```python
def agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_msg = messages[-1]
    
    # ToolMessage 处理
    if isinstance(last_msg, ToolMessage):
        system_prompt = """你是智能客服助手，根据工具结果自然回答用户。
要求：简洁、专业、直接回答用户问题。
工具结果："""
        # ✅ 正确：带上完整的工具结果
        llm_messages = [
            SystemMessage(content=system_prompt + last_msg.content),
            HumanMessage(content="请根据以上工具结果回答用户"),  # ✅ 简化，不要重复历史
        ]
        response = llm_with_tools.invoke(llm_messages)
        return {"messages": [response]}
```

**修改文件**: `src/services/agent.py`

---

#### B003: 实现真正的流式输出

**问题**: 当前只是 `yield new_content` 拼接，不是真正的流式

**修复方案**:
```python
def run_agent(session_id: str, user_input: str) -> Generator[str, None, None]:
    # ...
    for event in get_agent_graph().stream(initial_state):
        # 找到 AI 消息节点
        for node_name in ["rag", "agent"]:
            if node_name in event:
                messages = event[node_name].get("messages", [])
                if messages:
                    msg = messages[-1]
                    # ✅ 使用 response_iterator 实现 token 级流式
                    for chunk in llm_with_tools.stream(msg.content):
                        if chunk.content:
                            yield chunk.content
    
        # 处理工具节点
        if "tools" in event:
            tools_msg = event["tools"].get("messages", [])
            if tools_msg:
                yield f"[正在查询...]"
```

**修改文件**: `src/services/agent.py`

---

#### B004: 添加自动初始化

**问题**: 没有自动初始化知识库的逻辑

**修复方案**: 在 `app.py` 启动时检查并初始化

```python
# app.py 开头添加
from src.services.rag import get_rag, init_from_files

# 检查知识库
def init_knowledge_base():
    rag = get_rag()
    try:
        docs = rag.similarity_search("测试", k=1)
        if not docs:
            print("正在初始化知识库...")
            init_from_files()
    except:
        print("正在初始化知识库...")
        init_from_files()
```

**修改文件**: `app.py`

---

### Phase 2: 意图识别增强 ⚠️

#### I001: 改进意图识别

**问题**: 纯规则匹配，无语义理解能力

**修复方案**: 引入 embedding + 余弦相似度匹配

```python
# intent.py 新增
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 典型问句库
INTENT_EXAMPLES = {
    Intent.ORDER_QUERY: [
        "查一下我的订单", "订单到哪了", "我想看订单状态",
        "订单号是多少", "看看我买的电脑到哪了"
    ],
    Intent.LOGISTICS_QUERY: [
        "物流信息", "快递到哪了", "发货了吗",
        "什么时候到", "查一下物流"
    ],
    # ...
}

def recognize_intent(text: str) -> Intent:
    # 1. 规则匹配（优先）
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text.lower(), re.IGNORECASE):
                return intent
    
    # 2. 语义匹配（兜底）
    return semantic_match_intent(text)
```

**修改文件**: `src/services/intent.py`

---

### Phase 3: 部署优化 📝

#### 完善初始化脚本

**新增**: `scripts/init_all.py`

```python
#!/usr/bin/env python3
"""一键初始化脚本"""

def main():
    print("1. 初始化 PostgreSQL...")
    # 运行 SQL 脚本
    
    print("2. 初始化知识库...")
    from src.services.rag import init_from_files
    count = init_from_files()
    print(f"   已导入 {count} 个文档")
    
    print("3. 验证连接...")
    # 测试各服务
    
    print("初始化完成！")

if __name__ == "__main__":
    main()
```

---

## 三、实施顺序

```
Phase 1 (核心 Bug)
├── B001: RAG 初始化逻辑
├── B002: Agent 消息传递
├── B003: 流式输出
└── B004: 自动初始化

Phase 2 (意图增强)
└── I001: 语义意图识别

Phase 3 (部署优化)
├── 完善初始化脚本
├── 添加健康检查
└── 补充单元测试
```

---

## 四、验收标准

### Phase 1 完成后
- [ ] 多次启动不会丢失知识库数据
- [ ] 查询订单后，AI 能正确理解工具返回结果
- [ ] 流式输出有明显的逐字显示效果
- [ ] 首次启动自动初始化知识库

### Phase 2 完成后
- [ ] "我想查一下我的订单" 能正确识别为 ORDER_QUERY
- [ ] 模糊问句意图识别准确率提升

### Phase 3 完成后
- [ ] `python scripts/init_all.py` 一键初始化成功
- [ ] `GET /health` 返回服务状态
- [ ] 核心功能有单元测试覆盖