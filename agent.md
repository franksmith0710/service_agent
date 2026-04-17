### 核心必看内容                                     你是资深Agent开发工程师，熟悉LangGraph、LangChain等相关技术栈，按最新版LangChain（v1.0+）写法，产出规范、可运行、注释清晰的代码，不冗余、不遗漏核心功能。请按以下需求，开发企业级AI智能客服机器人，核心数据用占位符（标注清晰），规避已知问题，确保Demo级可直接运行。这是一个专门针对机械革命笔记本电脑的智能客服机器人

### 要求如下
#### 1. 界面：Streamlit做一个简单的聊天界面，支持用户输入
### 2. 技术栈：
   - 大模型：支持本地ollama模型的qwen3.5 4b 和硅基流动的qwen3.5 4b(OPENAI_API_KEY=sk-mkvwihbbwuvirgxkdfmonkonurkpucfvdwvbhcjqlswwpvbr
     OPENAI_BASE_URL=https://api.siliconflow.cn/v1)   把这个两个模型放置在一个文件里，只使用本地模型，我想改时我会手动改代码的
   - 框架：LangChain + LangGraph
   - 向量库：Chroma
   - 嵌入：本地ollama模型的bge-m3

### 3.可以用langsmith来进行监控和分析，（api_key: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）
### 4. 功能：
   - RAG 知识库问答
   - 工具调用：用户问题解答、查询订单、查询物流、转接人工
   - 多轮对话记忆，使用redis实现记忆存储
   - 严格不编造答案
   - 只回答客服问题
   上下文管理：状态管理
   控制流设计：流程编排
   错误恢复：异常处理
   反馈回路：监控告警
### 5.项目结构

```
kefu_agent/
├── app.py                      # Streamlit Web 入口
├── api.py                      # FastAPI 接口
├── src/
│   ├── config/
│   │   ├── settings.py          # 配置加载 (dataclass)
│   │   └── logger.py            # 日志配置
│   ├── models/
│   │   └── types.py             # 数据类型定义
│   └── services/
│       ├── agent.py             # Agent 核心
│       ├── llm.py               # LLM 管理
│       ├── tools.py             # 工具定义
│       ├── memory.py            # 对话记忆
│       ├── rag.py               # RAG 知识库
│       ├── database.py          # JSON 数据层
│       └── intent.py            # 意图识别
├── data/
│   └── mock_data.json           # 模拟数据
├── tests/
│   └── test_services.py         # 单元测试
├── .env                         # 环境变量
├── .env.example                 # 环境变量模板
├── .gitignore                   # Git 忽略配置
├── requirements.txt             # 依赖
└── README.md
```
### 6. Agent 流程图

```
用户输入 → 加载记忆(Redis) → intent_node(意图识别) → router(路由)

router 路由:
├─ chat (问候/感谢/再见) → chat_node → END
├─ product/pre_sales/after_sales → rag_node(RAG检索) → END  
├─ order/logistics/user → agent_node(ReAct循环)
│      ↓
│  tools_node(工具调用) → agent_node 循环
│      ↓
└─ transfer (转人工) → transfer_node → END

保存记忆 → END
```

### 意图识别

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
### 7.已知数据内容(自行添加到数据库或向量数据库)
 '''
 详细数据访问jxgm.md
 '''
### 6. 代码必须：
   - 核心真实数据可以先不写，等后续我会再添加可以使用占位符代替 。
   - 注释清晰
   - 最新版 LangChain 写法
### 7. 数据接入
真实数据参考“jxgm.md”



。