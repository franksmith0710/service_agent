import streamlit as st
import time
from typing import Generator

from agent import run_agent


st.set_page_config(page_title="智能客服", page_icon="🤖", layout="wide")


if "messages" not in st.session_state:
    st.session_state.messages = []


if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"


def stream_response(response: str) -> Generator[str, None, None]:
    for word in response:
        yield word
        time.sleep(0.02)


def main():
    st.title("🤖 智能客服")
    st.markdown("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                response = run_agent(st.session_state.session_id, prompt)

                for chunk in stream_response(response):
                    full_response += chunk
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                error_msg = f"抱歉，出现了一些问题：{str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

    with st.sidebar:
        st.header("操作")
        if st.button("清空对话"):
            st.session_state.messages = []
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
