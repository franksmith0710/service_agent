"""Streamlit 客服界面主程序"""
import streamlit as st
from agent import run_agent, init_rag, rag_retriever
from datetime import datetime


# 页面配置
st.set_page_config(
    page_title="智能客服",
    page_icon="💬",
    layout="wide"
)


# 初始化知识库
if "rag_initialized" not in st.session_state:
    init_rag()
    st.session_state.rag_initialized = True


# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []


# CSS 样式
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        text-align: right;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    .typing-indicator {
        display: inline-block;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)


def display_message(role: str, content: str):
    """显示消息"""
    with st.chat_message(role):
        st.markdown(content)


def get_model_choice():
    """获取模型选择"""
    return st.sidebar.selectbox(
        "选择模型",
        ["硅基流动 Qwen2.5-4B", "本地 Ollama Qwen2.5-4B"],
        index=0
    )


def main():
    """主函数"""
    st.title("🤖 智能客服助手")
    st.markdown("您好！我是智能客服，有什么可以帮助您的吗？")
    
    # 侧边栏
    with st.sidebar:
        st.header("设置")
        
        # 模型选择
        model_choice = get_model_choice()
        use_local = model_choice.startswith("本地")
        
        # 清空对话
        if st.button("清空对话"):
            st.session_state.messages = []
            st.rerun()
        
        # 显示知识库状态
        st.divider()
        st.subheader("知识库状态")
        if rag_retriever:
            st.success("✅ 已加载")
        else:
            st.warning("⚠️ 未加载")
        
        st.divider()
        st.caption("支持功能：FAQ查询、订单查询、物流查询、转接人工")


def run_agent_streaming(user_input: str):
    """流式运行 Agent
    
    Args:
        user_input: 用户输入
    
    Yields:
        流式输出的文本片段
    """
    from agent import agent
    from langchain_core.messages import HumanMessage, AIMessage
    
    # 构建消息历史
    messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # 添加当前输入
    messages.append(HumanMessage(content=user_input))
    
    # 流式输出
    full_response = ""
    for chunk in agent.stream({"messages": messages}):
        if "messages" in chunk:
            content = chunk["messages"][-1].content
            if content:
                full_response += content
                yield content


if __name__ == "__main__":
    main()
    
    # 显示历史消息
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        display_message("user", prompt)
        
        # 生成回复
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            try:
                # 流式输出
                for chunk in run_agent_streaming(prompt):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                
                # 最终响应（去掉光标）
                response_container.markdown(full_response)
                
            except Exception as e:
                error_msg = f"抱歉，出现了一些问题: {str(e)}"
                response_container.markdown(error_msg)
                full_response = error_msg
        
        # 添加助手消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat()
        })  