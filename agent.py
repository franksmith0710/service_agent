"""Agent 逻辑模块 - 使用 LangGraph 构建智能客服"""
import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from tools import AVAILABLE_TOOLS, search_knowledge_base, query_order, query_logistics, transfer_to_human


# 全局变量：RAG 检索器
rag_retriever = None


# LangSmith 配置
LANGSMITH_API_KEY = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "kefu-agent"


# 模型配置
class ModelConfig:
    """模型配置类"""
    
    # 硅基流动 API 配置
    SILICON_API_KEY = "sk-mkvwihbbwuvirgxkdfmonkonurkpucfvdwvbhcjqlswwpvbr"
    SILICON_BASE_URL = "https://api.siliconflow.cn/v1"
    SILICON_MODEL = "Qwen/Qwen2.5-4B-Instruct"
    
    # Ollama 本地配置
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5:4b"
    OLLAMA_EMBED_MODEL = "bge-m3"


def get_llm(use_local: bool = False):
    """获取 LLM 实例
    
    Args:
        use_local: 是否使用本地 Ollama 模型，False 使用硅基流动
    
    Returns:
        LLM 实例
    """
    if use_local:
        return ChatOllama(
            base_url=ModelConfig.OLLAMA_BASE_URL,
            model=ModelConfig.OLLAMA_MODEL,
            temperature=0.7,
            streaming=True
        )
    else:
        return ChatOpenAI(
            api_key=ModelConfig.SILICON_API_KEY,
            base_url=ModelConfig.SILICON_BASE_URL,
            model=ModelConfig.SILICON_MODEL,
            temperature=0.7,
            streaming=True
        )


class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[Sequence[BaseMessage], "对话历史"]


# System prompt
SYSTEM_PROMPT = """你是一个智能客服助手，专门帮助用户解答问题和处理请求。

## 重要约束
1. 只回答与客服相关的问题，不要回答与客服无关的问题
2. 严格不编造答案，不确定的问题要如实说明
3. 如果知识库没有相关信息，明确告知用户
4. 需要时使用工具查询真实信息，不要猜测

## 可用工具
- search_knowledge_base: 搜索知识库 FAQ
- query_order: 查询订单信息（需要提供订单号）
- query_logistics: 查询物流信息（需要提供快递单号）
- transfer_to_human: 转接人工客服

## 回复要求
- 语气友好、专业
- 回答简洁明了
- 如果需要查询订单或物流，请明确告知用户需要提供订单号/快递单号
- 如果问题无法解决，主动建议转接人工客服"""


# 初始化 LLM 和绑定工具
llm = get_llm(use_local=False)
llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)


def should_continue(state: AgentState) -> bool:
    """判断是否继续调用工具
    
    Returns:
        "continue" 表示需要继续执行工具，"end" 表示直接响应
    """
    last_message = state["messages"][-1]
    
    # 如果最后一条消息有工具调用，继续执行工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    
    # 如果最后一条消息是工具返回结果，也需要继续让 LLM 生成最终响应
    if hasattr(last_message, "type") and last_message.type == "tool":
        return "continue"
    
    return "end"


def call_model(state: AgentState) -> AgentState:
    """调用 LLM 生成响应
    
    Args:
        state: 当前状态
    
    Returns:
        更新后的状态
    """
    messages = state["messages"]
    
    # 构建消息列表，插入 system prompt
    if not any(isinstance(m, str) and "system" in str(type(m).__name__).lower() for m in messages):
        full_messages = [HumanMessage(content=SYSTEM_PROMPT)] + list(messages)
    else:
        full_messages = list(messages)
    
    # 调用 LLM
    response = llm_with_tools.invoke(full_messages)
    
    return {"messages": [response]}


def create_agent() -> StateGraph:
    """创建 Agent 图
    
    Returns:
        编译后的 Agent 图
    """
    # 创建图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", ToolNode(AVAILABLE_TOOLS))
    
    # 设置入口
    workflow.set_entry_point("llm")
    
    # 添加边
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # 工具执行后返回 LLM
    workflow.add_edge("tools", "llm")
    
    # 编译图
    return workflow.compile()


# 创建 Agent 实例
agent = create_agent()


def init_rag(knowledge_path: str = "knowledge"):
    """初始化 RAG 知识库
    
    Args:
        knowledge_path: 知识库文件路径
    """
    global rag_retriever
    
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader, MarkdownLoader
        
        # 嵌入模型
        embedding_model = OllamaEmbeddings(
            base_url=ModelConfig.OLLAMA_BASE_URL,
            model=ModelConfig.OLLAMA_EMBED_MODEL
        )
        
        # 加载知识库文档
        documents = []
        if os.path.exists(knowledge_path):
            for filename in os.listdir(knowledge_path):
                filepath = os.path.join(knowledge_path, filename)
                if filename.endswith('.txt'):
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(filepath, encoding='utf-8')
                elif filename.endswith('.md'):
                    from langchain_community.document_loaders import MarkdownLoader
                    loader = MarkdownLoader(filepath)
                else:
                    continue
                
                documents.extend(loader.load())
        
        if not documents:
            print("未找到知识库文档")
            return
        
        # 文档分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        # 创建向量库
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name="kefu-knowledge"
        )
        
        rag_retriever = vectorstore.as_retriever(
            search_type="similarity",
            k=3
        )
        
        print(f"知识库初始化完成，共加载 {len(splits)} 个文档片段")
        
    except Exception as e:
        print(f"知识库初始化失败: {e}")
        import traceback
        traceback.print_exc()


def run_agent(user_input: str, history: list = None) -> str:
    """运行 Agent 处理用户输入
    
    Args:
        user_input: 用户输入
        history: 对话历史
    
    Returns:
        Agent 响应
    """
    # 构建消息历史
    messages = []
    
    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    
    # 添加当前输入
    messages.append(HumanMessage(content=user_input))
    
    # 运行 Agent
    result = agent.invoke({"messages": messages})
    
    # 返回最后一条消息
    return result["messages"][-1].content


if __name__ == "__main__":
    # 测试
    init_rag()
    response = run_agent("你好，我想咨询一下订单问题")
    print(response)