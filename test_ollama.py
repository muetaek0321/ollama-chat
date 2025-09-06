
# from langchain_ollama.llms import OllamaLLM

# message = "Ollamaについて200字程度で簡潔に教えて"
# llm = OllamaLLM(model='gpt-oss:20b')
# response = llm.invoke(message)

# print(type(response))
# print(response)


from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Ollamaについて200字程度で簡潔に教えて")]
llm = ChatOllama(model='gpt-oss:20b')
response = llm.invoke(messages)

print(type(response))
print(response)
print(response.content)


# from langchain_ollama.llms import OllamaLLM
# from langchain_core.messages import HumanMessage

# messages = [HumanMessage(content="Ollamaについて200字程度で簡潔に教えて")]
# llm = OllamaLLM(model='gpt-oss:20b')
# response = llm.invoke(messages)

# print(type(response))
# print(response)