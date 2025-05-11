from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from typing import Annotated, Dict, List, Any, TypedDict, Union
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough

# 1. Define models
generation_model = ChatOllama(model="llama3.2:latest", temperature=0.7)
fact_check_model = ChatOllama(model="phi4:latest", temperature=0.0)
agent_llm = ChatOllama(model="phi4:latest", temperature=0.5)

# 2. Define State
class State(TypedDict):
    # Union is a type that can hold one of several types,
    # meaning it can be either a HumanMessage or an AIMessage
    messages: List[Union[HumanMessage, AIMessage | FunctionMessage]]
    content: str
    num_iterations: int  # Add a counter to track iterations

# 3. Define the tool
@tool
def web_search_tool(query: str) -> str:
    """Conduct a web search based on the provided query."""
    # Use the DuckDuckGoSearchResults with full API (not 'lite').
    wrapper = DuckDuckGoSearchAPIWrapper()
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    search_results = search.run(query)
    return search_results

# 4. Define nodes
def generate_content(state: State) -> Dict:
    messages = state['messages']
    response = generation_model.invoke(messages)
    new_message = AIMessage(content=response.content)
    return {"messages": messages + [new_message], "content": response.content, "num_iterations": state['num_iterations'] + 1}

def check_factual_accuracy(state: State) -> Dict:
    content = state['content']
    prompt = f"Assess accuracy: {content}"
    response = fact_check_model.invoke(prompt)
    return {"assessment": response.content.lower().strip()}

def decide_if_accurate(state: State) -> str:
    # Add a new condition to prevent infinite loops: Stop after 3 iterations.
    if "accurate" in state.get("assessment", "") or state["num_iterations"] >= 3:
        return END
    return "generate"

def call_agent(state: State) -> Dict:
    """
    Call the agent to decide whether to search the web or not.
    If search is required, the agent will use the web_search_tool.
    """
    messages = state["messages"]
    tools = {"web_search_tool": web_search_tool}

    # Define the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant, that can use tools. You have access to the following tool:\n web_search_tool: Conduct a web search based on the provided query."),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "You must first decide if you need to use a tool. If you decide to use a tool, use this format: \n```\nThought: Do I need to use a tool? Yes\nAction: <tool_name>\nAction Input: <tool_input>\n```\nIf not, use this format: \n```\nThought: Do I need to use a tool? No\nFinal Answer: <your answer>\n```"),
        ]
    )
    agent = prompt | agent_llm

    # Get Agent response
    agent_response = agent.invoke({"messages": messages})

    # Check if the agent wants to use a tool
    if "Thought: Do I need to use a tool? Yes" in agent_response:
        # Extract tool name and input
        tool_name_start = agent_response.find("Action: ") + len("Action: ")
        tool_name_end = agent_response.find("\n", tool_name_start)
        tool_name = agent_response[tool_name_start:tool_name_end].strip()
        
        tool_input_start = agent_response.find("Action Input: ") + len("Action Input: ")
        tool_input = agent_response[tool_input_start:].strip()

        # Invoke the tool
        if tool_name in tools:
            tool_result = tools[tool_name].invoke(tool_input)
        else:
            tool_result = f"Error: Tool '{tool_name}' not found."
        
        agent_response = f"Tool Result: {tool_result}"
    elif "Thought: Do I need to use a tool? No" in agent_response:
        final_answer_start = agent_response.find("Final Answer: ") + len("Final Answer: ")
        final_answer = agent_response[final_answer_start:].strip()
        agent_response = f"No web search needed: {final_answer}"
    else:
        agent_response = "Error: Could not determine if a tool should be used or not."

    # Return agent response
    agent_response = AIMessage(content=agent_response)

    return {"messages": messages + [agent_response], "num_iterations": state['num_iterations']}

def incorporate_search_results(state: State) -> Dict:
    """Incorporate web search results into the content."""
    messages = state['messages']
    last_message = messages[-1].content

    content = state.get("content")

    if "No web search needed" not in last_message:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant, that has access to internet search results."),
            ("human", f"Refine the following content based on the web search results provided.  Here is the original content: {content}. \n\nHere are the web search results: {last_message}")
        ])

        chain = prompt | generation_model

        response = chain.invoke({})
        new_message = AIMessage(content=response.content)
        return {"messages": messages + [new_message], "content": response.content, "num_iterations": state['num_iterations']}
    else:
        return {"num_iterations": state['num_iterations']}  # if no search results, nothing to incorporate, so just pass state along

def decide_agent_or_fact_check(state: State) -> str:
    """Decide whether to call the agent or go to fact-checking."""
    last_message = state["messages"][-1].content
    if "No web search needed" in last_message:
        return "fact_check"
    else:
        return "incorporate"

# 5. Build graph
graph = StateGraph(State)
graph.add_node("generate", generate_content)
graph.add_node("fact_check", check_factual_accuracy)
graph.add_node("call_agent", call_agent)
graph.add_node("incorporate", incorporate_search_results)

# Define conditional edges
graph.add_conditional_edges("fact_check", decide_if_accurate, {"generate": "generate", END: END})
graph.add_conditional_edges("call_agent", decide_agent_or_fact_check, {"incorporate": "incorporate", "fact_check": "fact_check"})
graph.add_edge("generate", "call_agent")
graph.add_edge("incorporate", "generate")
graph.set_entry_point("generate")

runnable = graph.compile()

# 6. Execute with proper input format
inputs = {"messages": [HumanMessage(content="Write a 1,000-word blog post about the effects of fasting on the human body.")], "num_iterations": 0}
result = runnable.invoke(inputs)

print("Final content:", result["content"])

# Save the final content as "Untitled.txt" in the current directory.
with open("Untitled.md", "w") as f:
    f.write(result["content"])
