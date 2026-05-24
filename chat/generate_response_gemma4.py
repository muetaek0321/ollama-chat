import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./pretrained" # 事前学習モデルの保存先指定

import streamlit as st
from markdown import Markdown
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoProcessor


# 環境変数設定
load_dotenv()

__all__ = [
    "response_generator_langchain_gemma4_rag",
]


RAG_PROMPT = """
あなたは質問応答タスクのアシスタントです。
検索された以下のコンテキストの一部を使って質問に丁寧に答えてください。
答えがわからなければ、わからないと答えてください。

質問: {question}
コンテキスト: {context}
答え:
"""
DATABASE_DIR = "./database/chroma"

# Target Model
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
target_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it",
    dtype="auto",
    device_map="auto",
)

# Assistant Model (the drafter)
assistant_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it-assistant",
    dtype="auto",
    device_map="auto",
)

def response_generator_langchain_gemma4_rag() -> str:
    """langchain_huggingfaceを使用+RAG（Gemma4用）
    """    
    # 直前のユーザの入力を取得
    user_input = st.session_state.messages[-1]["content"]
    
    with open("./biography_context.txt", mode="r", encoding="utf-8") as f:
        context = f.read()
    
    # システムプロンプトの用意
    system_prompt = [{
        "role": "system",
        "content": "あなたはユーザの質問に答えるアシスタントです。"
    }]
    
    # 直前のユーザの入力を取得
    rag_input = [{
        "role": "user",
        "content": RAG_PROMPT.format(question=user_input, context=context)
    }]
    
    # Process input
    text = processor.apply_chat_template(
        system_prompt + st.session_state.messages[:-1] + rag_input, 
        tokenize=False, 
        add_generation_prompt=True, 
    )
    inputs = processor(text=text, return_tensors="pt").to(target_model.device)
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate output
    outputs = target_model.generate(
        **inputs,
        assistant_model=assistant_model,
        max_new_tokens=1024,
    )
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    
    # Parse output
    parsed_response = processor.parse_response(response)
    
    response_html = Markdown().convert(parsed_response["content"])
    
    return response_html
  

