import asyncio
import time
import json
from typing import Generator, List, Dict

import streamlit as st
from markdown import Markdown
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage


# 定数
MCP_PROMPT = """
あなたは質問応答タスクのアシスタントです。
登録されているツールを使って質問に答えてください。

質問: {question}
答え: {agent_scratchpad}
"""
MCP_CONFIG_PATH = "./chat/mcp_config.json"

ROLES = {
    "user": HumanMessage,
    "assistant": AIMessage
}


__all__ = ["response_generator_mcp"]


def response_generator_mcp() -> Generator:
    """AIが作成した返答をstreamで返す関数
    """    
    response = asyncio.run(create_gemini_mcp_response(st.session_state.messages))
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# AIの返答を作成
async def create_gemini_mcp_response(
    messages: List[Dict[str, str]]
) -> str:        
    # モデルを準備
    # llm = ChatOllama(model='gpt-oss:20b')
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    # プロンプトを設定
    prompt = PromptTemplate.from_template(MCP_PROMPT)
    
    # 直前のユーザの入力を取得
    user_input = messages[-1]["content"]
    
    # MCPサーバの設定を読み込み
    with open(MCP_CONFIG_PATH, mode="r") as f:
        mcp_config = json.load(f)
    
    # ツール化
    mcp_client = MultiServerMCPClient(mcp_config["mcpServers"])
    tools = await mcp_client.get_tools()
    
    # エージェントを用意
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    
    # 返答を取得
    response = await executor.ainvoke({"question": user_input})
    
    # 返答を成形（makrddown -> HTML）
    response = Markdown().convert(response["output"])
    
    return response
    
