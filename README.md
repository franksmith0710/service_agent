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
只作为一个参考，不必完全按照这个结构来，根据你的需求来组织代码。
```
vertu_sales_agent/
├── app/                        # 核心应用目录
│   ├── app.py                  # FastAPI 应用工厂
│   ├── config.py               # 全局配置
│   ├── core/                   # 核心公共模块
│   │   ├── shared.py           # 核心共享逻辑
│   │   └── middlewares.py      # 中间件
│   └── services/               # 服务模块目录
│       ├── __init__.py
│       ├── react_agent/        
│       │   ├── agent.py        # 核心 ReAct 代理
│       │   ├── prompts.py      # 系统提示词
│       │   ├── tools.py       # 工具实现
│       │   └── （其余文件模块和内容自行补充）
├── main.py                     # 应用入口点
├── requirements.txt            # pip 依赖
```
结构不定可以按需添加和修改其他文件
### 6. 代码必须：
   - 核心真实数据可以先不写，等后续我会再添加可以使用模拟数据代替，后续更改。
   - 注释清晰
   - 使用最新版 LangChain 写法
### 7. 内容不需要很全只做一个小demo试试效果先完成这些功能，其余内容等我安排，不要被我限制，自行发挥


### 8.认真检查，把以上功能全部完成之后进入下一步