import logging
import os
from langchain_core.messages import HumanMessage, AIMessage
from config import LLM_CONFIG, SILICONFLOW_CONFIG, LANGSMITH_CONFIG
from tools import query_order, query_logistics, transfer_to_human
from rag import create_knowledge_base
from memory import get_memory

logging.basicConfig(level=logging.WARNING)

langsmith_key = LANGSMITH_CONFIG.get("api_key") or ""
os.environ["LANGSMITH_API_KEY"] = langsmith_key
os.environ["LANGSMITH_TRACING_V2"] = "true"

llm_model = LLM_CONFIG["model"]
llm_provider = LLM_CONFIG.get("provider", "ollama")


def get_llm():
    """获取 LLM 实例，根据配置选择模型"""
    if llm_provider == "siliconflow":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=SILICONFLOW_CONFIG["model"],
            base_url=SILICONFLOW_CONFIG["base_url"],
            api_key=SILICONFLOW_CONFIG["api_key"],
            temperature=LLM_CONFIG["temperature"],
        )
    else:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=LLM_CONFIG["model"],
            base_url=LLM_CONFIG["base_url"],
            temperature=LLM_CONFIG["temperature"],
        )


def get_rag():
    """获取 RAG 实例"""
    return create_knowledge_base()


SYSTEM_PROMPT = """你是一个智能客服助手。请严格遵守以下规则：
1. 只能回答与客服相关的问题，不要回答无关问题
2. 如果不知道答案，直接说不知道，不要编造
3. 如果需要查询订单或物流，请使用工具
4. 如果用户要求转接人工，使用工具
5. 使用知识库回答常见问题
6. 保持专业、礼貌的回答风格
"""

TOOLS = [query_order, query_logistics, transfer_to_human]


def create_agent_graph():
    """使用 LangChain 新版 create_agent"""
    from langchain.agents import create_agent

    llm = get_llm()
    agent = create_agent(
        llm,
        TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


agent_graph = create_agent_graph()


rag_instance = None


def get_rag_instance():
    global rag_instance
    if rag_instance is None:
        rag_instance = get_rag()
    return rag_instance


def run_agent(session_id: str, user_input: str):
    """
    执行 Agent 处理用户输入，支持流式输出
    """
    memory = get_memory(session_id)
    history = memory.get_messages()

    try:
        rag = get_rag_instance()
        docs = rag.similarity_search(user_input, k=3)
        context = "\n".join([d.page_content for d in docs]) if docs else ""
    except Exception as e:
        logging.warning(f"RAG search failed: {e}")
        context = ""

    if context:
        enhanced_input = f"{user_input}\n\n相关知识：{context}"
    else:
        enhanced_input = user_input

    messages = list(history) + [HumanMessage(content=enhanced_input)]

    full_response = ""
    for chunk in agent_graph.stream({"messages": messages}):
        if "messages" in chunk:
            content = chunk["messages"][-1].content
            if content:
                new_content = content[len(full_response) :]
                full_response = content
                yield new_content

    memory.add_user_message(user_input)
    memory.add_ai_message(full_response)
