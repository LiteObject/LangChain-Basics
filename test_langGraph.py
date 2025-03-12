from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import Annotated, Dict, List, Any, TypedDict, Union

# 1. Define models
generation_model = ChatOllama(model="llama3.2:latest", temperature=0.7)
fact_check_model = ChatOllama(model="phi4:latest", temperature=0.0)

# 2. Define State
class State(TypedDict):

    # Union is a type that can hold one of several types,
    # meaning it can be either a HumanMessage or an AIMessage
    messages: List[Union[HumanMessage, AIMessage]]
    content: str

# 3. Define nodes
def generate_content(state: State) -> Dict:
    messages = state['messages']
    response = generation_model.invoke(messages)
    new_message = AIMessage(content=response.content)
    return {"messages": messages + [new_message], "content": response.content}

def check_factual_accuracy(state: State) -> Dict:
    content = state['content']
    prompt = f"Assess accuracy: {content}"
    response = fact_check_model.invoke(prompt)
    return {"assessment": response.content.lower().strip()}

# 4. Build graph
graph = StateGraph(State)
graph.add_node("generate", generate_content)
graph.add_node("fact_check", check_factual_accuracy)

def decide_if_accurate(state: State) -> str:
    if "accurate" in state.get("assessment", ""):
        return END
    return "generate"

graph.add_conditional_edges("fact_check", decide_if_accurate, {"generate": "generate", END: END})
graph.add_edge("generate", "fact_check")
graph.set_entry_point("generate")

runnable = graph.compile()

# 5. Execute with proper input format
inputs = {"messages": [HumanMessage(content="Write a 1,000-word blog post about the effects of fasting on the human body.")]}
result = runnable.invoke(inputs)

print("Final content:", result["content"])

# Save the final content as "Untitled.txt" in the current directory.
with open("Untitled.md", "w") as f:
    f.write(result["content"])
