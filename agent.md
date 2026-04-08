帮我生成一个完整可直接运行的 Python 智能客服项目，使用 Streamlit 做前端界面。

要求：
1. 界面：Streamlit 聊天界面，支持用户输入、对话历史、流式输出（打字机效果）
2. 技术栈：
   - 大模型：支持本地ollama模型的qwen3.5 4b 和硅基流动的qwen3.5 4b(OPENAI_API_KEY=sk-mkvwihbbwuvirgxkdfmonkonurkpucfvdwvbhcjqlswwpvbr
OPENAI_BASE_URL=https://api.siliconflow.cn/v1)
   - 框架：LangChain + LangGraph
   - 向量库：Chroma
   - 嵌入：本地ollama模型的bge-m3

还可以用langsmith来进行监控和分析（api_key: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）
3. 功能：
   - RAG 知识库问答
   - 工具调用：用户问题解答、查询订单、查询物流、转接人工
   - 多轮对话记忆
   - 严格不编造答案
   - 只回答客服问题
4. 项目结构：
   app.py              # 主程序（Streamlit）
   agent.py            # Agent 逻辑
   tools.py            # 工具函数
   knowledge/          # 知识库
   requirements.txt
   结构不定可以按需添加其他文件
5. 代码必须：
   - 核心真实数据可以先不写，等后续再添加。先把主要框架实现。
   - 注释清晰
   - 最新版 LangChain 写法
   - 不需要我写任何前端
6. 最后给我启动命令


---

## 完成进度

### ✅ 已完成
1. **requirements.txt** - 依赖配置文件
2. **tools.py** - 工具函数模块
   - search_knowledge_base: 搜索知识库
   - query_order: 查询订单（模拟数据）
   - query_logistics: 查询物流（模拟数据）
   - transfer_to_human: 转接人工
3. **agent.py** - Agent 逻辑模块
   - LangGraph 状态图构建
   - 支持硅基流动/本地 Ollama 切换
   - RAG 知识库集成（Chroma + bge-m3）
   - LangSmith 监控集成
4. **app.py** - Streamlit 主程序
   - 聊天界面
   - 流式输出（打字机效果）
   - 对话历史
   - 模型切换
5. **knowledge/** - 知识库目录
   - faq.md: 常见问题
   - products.md: 产品信息

### 🔧 启动命令

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 确保 Ollama 已启动并安装了 bge-m3 嵌入模型
# ollama pull bge-m3

# 3. 启动 Streamlit
streamlit run app.py
```

### ⚠️ 注意事项
- 首次启动时知识库会自动加载 knowledge/ 目录下的文档
- 可在侧边栏切换使用硅基流动 API 或本地 Ollama
- 订单和物流查询使用模拟数据，需后续接入真实系统