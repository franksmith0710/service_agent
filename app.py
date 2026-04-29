"""
智能客服 Web 界面

基于 Streamlit 的聊天界面
"""

import streamlit as st
from langchain_core.messages import HumanMessage

from src.services.agent import run_agent
from src.services.memory import get_memory
from src.services.rag import init_from_files, get_rag
from src.config.settings import config
from src.config.logger import setup_logger

logger = setup_logger(name="kefu_agent", level=20)


def init_knowledge_base():
    """检查并初始化知识库（带缓存）"""
    if hasattr(st, "session_state") and "kb_initialized" in st.session_state:
        return

    try:
        rag = get_rag()
        docs = rag.similarity_search("测试", k=1)
        if not docs:
            print("知识库为空，正在初始化...")
            init_from_files()
            print("知识库初始化完成")
        if hasattr(st, "session_state"):
            st.session_state.kb_initialized = True
    except Exception as e:
        print(f"初始化知识库失败: {e}")
        if hasattr(st, "session_state"):
            st.session_state.kb_initialized = True  # 即使失败也标记，避免重复尝试


# 设置页面配置
st.set_page_config(
    page_title="智能客服",
    page_icon="🤖",
    layout="wide",
)

# 初始化会话 ID
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"


def main():
    """主函数"""
    with st.spinner("正在初始化知识库..."):
        init_knowledge_base()

    st.title("🤖 智能客服")
    st.markdown("---")

    # 显示对话历史
    memory = get_memory(st.session_state.session_id)
    history = memory.get_messages()
    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner("正在思考中..."):
                    for chunk in run_agent(st.session_state.session_id, prompt):
                        full_response += chunk
                        message_placeholder.markdown(full_response)

            except Exception as e:
                logger.error(f"Agent error: {e}")
                error_msg = f"抱歉，出现了一些问题：{str(e)}"
                message_placeholder.markdown(error_msg)

    # 侧边栏
    with st.sidebar:
        st.header("操作")
        if st.button("清空对话"):
            from src.services.memory import clear_memory

            clear_memory(st.session_state.session_id)
            st.rerun()

        st.markdown("---")
        st.header("说明")
        st.markdown("""
        - 输入商品相关问题进行咨询
        - 输入订单号查询订单
        - 输入快递单号查询物流
        - 输入"转人工"转接客服
        """)

        st.markdown("---")
        st.header("配置信息")
        st.markdown(f"- 调度模型: `{config.siliconflow.dispatch_model}`")
        st.markdown(f"- 生成模型: `{config.siliconflow.model}`")
        st.markdown(f"- 知识库: `{config.chroma.persist_directory}`")


if __name__ == "__main__":
    main()
