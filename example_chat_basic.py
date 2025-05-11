from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

llm = ChatOllama(model="phi4", temperature=0.5)
# print(llm.invoke("Hello"))

messages = [
    ("system", "You are an experienced stock market analyst."),
    ("user", "What are the top 5 stock symbols?"),
]

print(llm.invoke(messages))
