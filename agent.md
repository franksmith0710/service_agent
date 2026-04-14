### 核心必看内容                                     你是资深Agent开发工程师，熟悉LangGraph、LangChain等相关技术栈，按最新版LangChain（v1.0+）写法，产出规范、可运行、注释清晰的代码，不冗余、不遗漏核心功能。请按以下需求，开发企业级AI智能客服机器人，核心数据用占位符（标注清晰），规避已知问题，确保Demo级可直接运行

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
### 7. 接入真实数据库（自行编写，内容真实，数据丰富），包括订单、物流、用户信息等

### 8.完成多路召回，召回准确率要大于0.8，召回的向量库使用chroma数据库



。
