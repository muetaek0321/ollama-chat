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
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat.parse_json import parse_chat_output


# 環境変数設定
load_dotenv()

# 定数
# OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_MODEL = "gemma4:e4b"
HF_MODEL = "llm-jp/llm-jp-4-8b-thinking"
ROLES = {
    "user": HumanMessage,
    "assistant": AIMessage
}


__all__ = ["response_generator_ollama_python", 
           "response_generator_huggingface_model",
           "response_generator_langchain_ollama",
           "response_generator_langchain_huggingface",
           "response_generator_langchain_ollama_rag", 
           "response_generator_langchain_gemini_rag"]


def response_generator_ollama_python() -> str:
    """ollama-pythonを使用
    """
    response: ChatResponse = chat(
        model=OLLAMA_MODEL, 
        messages=st.session_state.messages,
        options={"temperature": 1, "reasoning_effort": "low"}
    )
    
    response_html = Markdown().convert(response.message.content)
    
    return response_html


def response_generator_huggingface_model() -> str:
    """HuggingFaceのモデルを使用
    """
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    system_prompt = [{
        "role": "system",
        "content": "あなたはユーザの質問に答えるアシスタントです。回答は200文字程度で要点だけをまとめて簡潔に答えてください。"
    }]
    
    prompt: str = tokenizer.apply_chat_template(
        system_prompt + st.session_state.messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",  # {"low", "medium", "high"}
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_tensor = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 出力データを変換して返答文を取得
    generated_ids: list[int] = output_tensor[0][inputs["input_ids"].shape[1]:].tolist()
    response = tokenizer.decode(generated_ids)
    response = parse_chat_output(response)
    
    response_html = Markdown().convert(response["assistant"]["message"])
    
    return response_html


def response_generator_langchain_ollama() -> str:
    """langchain_ollamaを使用
    """
    llm = OllamaLLM(model=OLLAMA_MODEL)
    
    system_prompt = "あなたはユーザの質問に答えるアシスタントです。回答は200文字程度で要点だけをまとめて簡潔に答えてください。"
    messages = [SystemMessage(content=system_prompt)] + [
        ROLES[msg["role"]](content=msg["content"]) 
        for msg in st.session_state.messages
    ]
    
    response = llm.invoke(messages)
    
    # with open("./output.md", mode="w", encoding="cp932") as f:
    #     f.write(response)
    
    response_html = Markdown().convert(response)
    
    return response_html


def response_generator_langchain_huggingface() -> str:
    """langchain_huggingfaceを使用
    """
    # llm = HuggingFaceEndpoint(
    #     endpoint_url=HF_MODEL,
    #     max_new_tokens=1024,
    #     do_sample=False,
    # )
    # llm = ChatHuggingFace(llm=llm)
    
    # tokenizer = AutoTokenizer.from_pretrained(
    #     HF_MODEL,
    #     # trust_remote_code is required to load custom tokenizer and reasoning parser.
    #     trust_remote_code=True,
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     HF_MODEL,
    #     dtype=torch.bfloat16,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )
    # model.eval()
    
    llm = HuggingFacePipeline(
        
    )
    
    system_prompt = "あなたはユーザの質問に答えるアシスタントです。回答は200文字程度で要点だけをまとめて簡潔に答えてください。"
    messages = [SystemMessage(content=system_prompt)] + [
        ROLES[msg["role"]](content=msg["content"]) 
        for msg in st.session_state.messages
    ]
    
    response = llm.invoke(messages)
    
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
    llm = OllamaLLM(model=OLLAMA_MODEL)
    
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
