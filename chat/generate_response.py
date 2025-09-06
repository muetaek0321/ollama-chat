import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./pretrained" # 事前学習モデルの保存先指定

import streamlit as st
import torch
from markdown import Markdown
from dotenv import load_dotenv
from ollama import chat
from ollama import ChatResponse
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 環境変数設定
load_dotenv()


__all__ = ["response_generator_ollama_python", "response_generator_langchain_ollama",
           "response_generator_langchain_ollama_rag", "response_generator_langchain_gemini_rag"]


def response_generator_ollama_python() -> str:
    """ollama-pythonを使用
    """
    response: ChatResponse = chat(
        model='gpt-oss:20b', 
        messages=st.session_state.messages,
        options={"temperature": 1, "reasoning_effort": "low"}
    )
    
    response_html = Markdown().convert(response.message.content)
    
    return response_html


ROLES = {
    "user": HumanMessage,
    "assistant": AIMessage
}


def response_generator_langchain_ollama() -> str:
    """langchain_ollamaを使用
    """
    llm = OllamaLLM(model='gpt-oss:20b')
    
    system_prompt = "あなたはユーザの質問に答えるアシスタントです。回答は200文字程度で要点だけをまとめて簡潔に答えてください。"
    messages = [SystemMessage(content=system_prompt)] + [
        ROLES[msg["role"]](content=msg["content"]) 
        for msg in st.session_state.messages
    ]
    
    response = llm.invoke(messages)
    
    with open("./output.md", mode="w", encoding="cp932") as f:
        f.write(response)
    
    response_html = Markdown().convert(response)
    
    return response_html
    

RAG_PROMPT = """
あなたは質問応答タスクのアシスタントです。
検索された以下のコンテキストの一部を使って質問に丁寧に答えてください。
答えがわからなければ、わからないと答えてください。

質問: {question}
コンテキスト: {context}
答え:
"""
DATABASE_DIR = "./database/chroma"

def response_generator_langchain_ollama_rag() -> str:
    """langchain_ollamaを使用+RAG
    """
    llm = OllamaLLM(model='gpt-oss:20b')
    
    # 直前のユーザの入力を取得
    user_input = st.session_state.messages[-1]["content"]
    
    # ベクトル化する準備
    model_kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu", 
        "trust_remote_code": True
    }
    embedding = HuggingFaceEmbeddings(
        model_name="pfnet/plamo-embedding-1b",
        model_kwargs=model_kwargs
    )
    
    # DBを読み込んで知識データ取得
    vectorstore = Chroma(collection_name="elephants", 
                         persist_directory=DATABASE_DIR, 
                         embedding_function=embedding)
    docs = vectorstore.similarity_search(query=user_input, k=10)
    context = "\n".join([f"Content:\n{doc.page_content}" for doc in docs])
    
    messages = [
        ROLES[msg["role"]](content=msg["content"]) 
        for msg in st.session_state.messages[:-1]
    ] + [HumanMessage(content=RAG_PROMPT.format(question=user_input, context=context))]
    
    response = llm.invoke(messages)
    
    with open("./output.md", mode="w", encoding="utf-8") as f:
        f.write(response)
    
    response_html = Markdown().convert(response)
    
    return response_html


def response_generator_langchain_gemini_rag() -> str:
    """Geminiを使用+RAG
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # 直前のユーザの入力を取得
    user_input = st.session_state.messages[-1]["content"]
    
    # ベクトル化する準備
    model_kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu", 
        "trust_remote_code": True
    }
    embedding = HuggingFaceEmbeddings(
        model_name="pfnet/plamo-embedding-1b",
        model_kwargs=model_kwargs
    )
    
    # DBを読み込んで知識データ取得
    vectorstore = Chroma(collection_name="elephants", 
                         persist_directory="chat/chroma", 
                         embedding_function=embedding)
    docs = vectorstore.similarity_search(query=user_input, k=10)
    context = "\n".join([f"Content:\n{doc.page_content}" for doc in docs])
    
    messages = [
        ROLES[msg["role"]](content=msg["content"]) 
        for msg in st.session_state.messages[:-1]
    ] + [HumanMessage(content=RAG_PROMPT.format(question=user_input, context=context))]
    
    response = llm.invoke(messages).content
    
    with open("./output.md", mode="w", encoding="cp932") as f:
        f.write(response)
    
    response_html = Markdown().convert(response)
    
    return response_html
