### 核心必看内容

### 你是一个agent开发工程师，你需要制作一个智能体客服的企业级项目，可以实现：多轮对话，用户关于商品的问题解答、查询订单、查询物流、转接人工等功能。是一个基于 FastAPI 和 LangGraph 构建的 AI 智能客服聊天机器人，采用 ReAct（推理+行动）代理模式。

### 要求如下
#### 1. 界面：Streamlit做一个简单的聊天界面，支持用户输入、流式输出（打字机效果）
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
### 5.项目结构

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
│       ├── agent.py             # Agent 核心
│       ├── llm.py             # LLM 管理
│       ├── tools.py             # 工具定义
│       ├── memory.py            # 对话记忆
│       └── rag.py             # RAG 知识库
├── tests/
│   └── test_services.py         # 单元测试
├── .env                         # 环境变量
├── .env.example                 # 环境变量模板
├── requirements.txt             # 依赖
└── README.md
```
### 6. 代码必须：
   - 核心真实数据可以先不写，等后续我会再添加可以使用占位符代替 。
   - 注释清晰
   - 最新版 LangChain 写法
### 7. 内容不需要很全只做一个小demo试试效果先完成这些功能，其余内容等我安排，不要被我限制，自行发挥


### 8.接入真实数据库，包括订单、物流、用户信息等