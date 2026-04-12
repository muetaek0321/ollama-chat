"""
https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
"""
import time

import streamlit as st

from chat.generate_response import *
from chat.generate_response_mcp import *


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
        s_time = time.perf_counter()
        # response_text = response_generator_ollama_python()
        # response_text = response_generator_huggingface_model()
        # response_text = response_generator_langchain_ollama()
        # response_text = response_generator_langchain_huggingface()
        # response_text = response_generator_langchain_ollama_rag()
        response_text = response_generator_langchain_huggingface_rag()
        # response_text = response_generator_langchain_gemini_rag()
        # response_text = response_generator_mcp()
        
        # 返答生成にかかった時間を表示
        print(f"Elapsed time: {time.perf_counter() - s_time:.2f} seconds")
        
        # ストリームで表示
        for res in response_text:
            response += res
            res_container.markdown(response, unsafe_allow_html=True)
            
    # AIの返答を履歴に保存
    st.session_state.messages.append({"role": "assistant", "content": response})
    
