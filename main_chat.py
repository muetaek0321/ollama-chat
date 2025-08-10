"""
https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
"""
import streamlit as st

from chat.generate_response import *


st.title("Ollama chat")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# これまでのチャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 入力
if prompt := st.chat_input("What is up?"):
    # userの入力を履歴に保存
    st.session_state.messages.append({"role": "user", "content": prompt})
    # userの入力を表示
    with st.chat_message("user"):
        st.markdown(prompt)

    # 返答生成＋表示
    with st.chat_message("assistant"):
        response = ""
        res_container = st.empty()
        
        # 返答生成
        # response_text = response_generator_ollama_python()
        # response_text = response_generator_langchain_ollama()
        response_text = response_generator_langchain_ollama_rag()
        # response_text = response_generator_langchain_gemini_rag()
        
        # ストリームで表示
        for res in response_text:
            response += res
            res_container.markdown(response, unsafe_allow_html=True)
            
    # AIの返答を履歴に保存
    st.session_state.messages.append({"role": "assistant", "content": response})
    
