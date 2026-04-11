import streamlit as st
from agent import run_agent
from memory import get_memory
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="智能客服", page_icon="🤖", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"


def main():
    st.title("🤖 智能客服")
    st.markdown("---")

    memory = get_memory(st.session_state.session_id)
    history = memory.get_messages()
    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if prompt := st.chat_input("请输入您的问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in run_agent(st.session_state.session_id, prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response)

            except Exception as e:
                error_msg = f"抱歉，出现了一些问题：{str(e)}"
                message_placeholder.markdown(error_msg)

    with st.sidebar:
        st.header("操作")
        if st.button("清空对话"):
            from memory import clear_memory

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


if __name__ == "__main__":
    main()
