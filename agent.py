import os
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from config import LLM_CONFIG, LANGSMITH_CONFIG
from tools import query_order, query_logistics, transfer_to_human
from memory import ChatMemory, get_memory


os.environ["LANGSMITH_API_KEY"] = LANGSMITH_CONFIG["api_key"]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "add_messages"]
    session_id: str
    should_transfer: Optional[bool]
    tool_called: Optional[str]


llm: Optional[ChatOllama] = None


def get_llm() -> ChatOllama:
    global llm
    if llm is None:
        llm = ChatOllama(
            model=LLM_CONFIG["model"],
            base_url=LLM_CONFIG["base_url"],
            temperature=LLM_CONFIG["temperature"],
        )
    return llm


def create_tools():
    return [
        {"name": "query_order", "description": "查询订单信息", "func": query_order},
        {
            "name": "query_logistics",
            "description": "查询物流信息",
            "func": query_logistics,
        },
        {
            "name": "transfer_to_human",
            "description": "转接人工客服",
            "func": transfer_to_human,
        },
    ]


SYSTEM_PROMPT = """你是一个智能客服助手，请严格遵守以下规则：
1. 只能回答与客服相关的问题，不要回答无关问题
2. 如果不知道答案，直接说不知道，不要编造
3. 如果需要查询订单或物流，请使用工具
4. 如果用户要求转接人工，使用工具
5. 使用知识库回答常见问题
6. 保持专业、礼貌的回答风格
"""


def should_use_tools(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1].content.lower()

    keywords_order = ["订单", "订单号", "买的东西", "买了什么"]
    keywords_logistics = ["物流", "快递", "发货", "运输", "到哪了"]
    keywords_transfer = ["人工", "人工客服", "转人工", "转接人工"]

    for keyword in keywords_transfer:
        if keyword in last_message:
            return "transfer"

    for keyword in keywords_order:
        if keyword in last_message:
            return "query_order"

    for keyword in keywords_logistics:
        if keyword in last_message:
            return "query_logistics"

    return "respond"


def query_order_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1].content

    import re

    order_id_match = re.search(r"\d{10,}", last_message)

    if order_id_match:
        order_id = order_id_match.group()
    else:
        order_id = "default_order_id"

    result = query_order(order_id)

    response = f"根据查询结果，您的订单信息如下：{result}"

    return {"messages": [AIMessage(content=response)], "tool_called": "query_order"}


def query_logistics_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1].content

    import re

    order_id_match = re.search(r"\d{10,}", last_message)

    if order_id_match:
        order_id = order_id_match.group()
    else:
        order_id = "default_order_id"

    result = query_logistics(order_id)

    response = f"根据查询结果，您的物流信息如下：{result}"

    return {"messages": [AIMessage(content=response)], "tool_called": "query_logistics"}


def transfer_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    messages_history = "\n".join([m.content for m in messages[:-1]])

    result = transfer_to_human(
        reason="用户请求转接人工客服", conversation_summary=messages_history[:500]
    )

    response = f"已为您转接人工客服，{result}"

    return {
        "messages": [AIMessage(content=response)],
        "should_transfer": True,
        "tool_called": "transfer_to_human",
    }


def chat_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    session_id = state["session_id"]

    memory = get_memory(session_id)
    memory_variables = memory.load_memory_variables()

    if memory_variables.get("history"):
        history = "\n".join(
            [
                f"用户: {m.content}"
                if isinstance(m, HumanMessage)
                else f"客服: {m.content}"
                for m in memory.get_messages()
            ]
        )
        system_message = SystemMessage(
            content=f"{SYSTEM_PROMPT}\n\n对话历史：\n{history}"
        )
    else:
        system_message = SystemMessage(content=SYSTEM_PROMPT)

    llm_instance = get_llm()

    all_messages = [system_message] + messages

    response = llm_instance.invoke(all_messages)

    if not isinstance(response, BaseMessage):
        response = AIMessage(content=str(response))

    return {"messages": [response]}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("should_use_tools", lambda state: state)
    graph.add_node("query_order", query_order_node)
    graph.add_node("query_logistics", query_logistics_node)
    graph.add_node("transfer", transfer_node)
    graph.add_node("chat", chat_node)

    graph.set_entry_point("should_use_tools")

    graph.add_conditional_edges(
        "should_use_tools",
        lambda state: should_use_tools(state),
    )

    graph.add_edge("query_order", END)
    graph.add_edge("query_logistics", END)
    graph.add_edge("transfer", END)
    graph.add_edge("chat", END)

    return graph.compile()


agent_graph = build_graph()


def run_agent(session_id: str, user_input: str):
    memory = get_memory(session_id)
    memory.add_user_message(user_input)

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "session_id": session_id,
        "should_transfer": False,
        "tool_called": None,
    }

    result = agent_graph.invoke(initial_state)

    response = result["messages"][-1].content
    memory.add_ai_message(response)

    return response
