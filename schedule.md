# 项目进度

## 完成状态

| 模块 | 状态 | 说明 |
|------|------|------|
| config.py | ✅ 完成 | LLM、LangSmith、嵌入、Chroma配置 |
| tools.py | ✅ 完成 | 查询订单、查询物流、转接人工工具 |
| rag.py | ✅ 完成 | Chroma向量库 + bge-m3嵌入 |
| memory.py | ✅ 完成 | 多轮对话记忆 |
| agent.py | ✅ 完成 | LangGraph图和节点 |
| app.py | ✅ 完成 | Streamlit聊天界面 |
| requirements.txt | ✅ 完成 | 依赖清单 |

## 功能清单

- [x] Streamlit聊天界面，带打字机效果流式输出
- [x] 本地Ollama的qwen3.5:4b模型
- [x] LangChain + LangGraph框架
- [x] Chroma向量库 + bge-m3嵌入
- [x] LangSmith监控配置
- [x] RAG知识库问答
- [x] 工具调用（订单、物流、转人工）
- [x] 多轮对话记忆
- [x] 只回答客服相关问题，不编造答案

## 待办事项

- [ ] 集成真实订单API
- [ ] 集成真实物流API
- [ ] 完善知识库内容
- [ ] 添加更多客服场景
- [ ] 部署上线