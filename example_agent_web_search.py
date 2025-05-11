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
# Reduced temperature for more deterministic behavior in this example
agent_llm = ChatOllama(model="phi4:latest", temperature=0.2)

# 2. Define State
class State(TypedDict):
    # Union is a type that can hold one of several types,
    # meaning it can be either a HumanMessage or an AIMessage
    messages: List[Union[HumanMessage, AIMessage | FunctionMessage]]
    content: str  # Store the content that might need web searching
    search_results: str # Store the web search results
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
            ("system", "You are a helpful assistant, that can use tools. "
             "You have access to the following tool:"
             "\n web_search_tool: Conduct a web search based on the provided query."),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "You must first decide if you need to use a tool. "
             "If you decide to use a tool, use this format: \n```\nThought: "
             "Do I need to use a tool? Yes\nAction: <tool_name>\nAction Input: "
             "<tool_input>\n```\nIf not, use this format: \n```\nThought: "
             "Do I need to use a tool? No\nFinal Answer: <your answer>\n```"),
        ]
    )
    
    agent = prompt | agent_llm

    # Get Agent response
    agent_response = agent.invoke({"messages": messages})

    # Check if the agent wants to use a tool
    if "Thought: Do I need to use a tool? Yes" in agent_response.content:
        # Extract tool name and input
        tool_name_start = agent_response.content.find("Action: ") + len("Action: ")
        tool_name_end = agent_response.content.find("\n", tool_name_start)
        tool_name = agent_response.content[tool_name_start:tool_name_end].strip()
        
        tool_input_start = agent_response.content.find("Action Input: ") + len("Action Input: ")
        tool_input = agent_response.content[tool_input_start:].strip()

        # Invoke the tool
        if tool_name in tools:
            tool_result = tools[tool_name].invoke(tool_input)
        else:
            tool_result = f"Error: Tool '{tool_name}' not found."
        
        agent_response = f"Tool Result: {tool_result}"
        return {"messages": messages + [AIMessage(content=agent_response)], "content": tool_input, "search_results": tool_result, "num_iterations": state['num_iterations'] + 1}
    elif "Thought: Do I need to use a tool? No" in agent_response.content:
        final_answer_start = agent_response.content.find("Final Answer: ") + len("Final Answer: ")
        final_answer = agent_response.content[final_answer_start:].strip()
        agent_response = f"No web search needed: {final_answer}"
        return {"messages": messages + [AIMessage(content=agent_response)], "content": final_answer, "search_results": "", "num_iterations": state['num_iterations'] + 1}
    else:
        agent_response = "Error: Could not determine if a tool should be used or not."
        return {"messages": messages + [AIMessage(content=agent_response)], "content": agent_response, "search_results": "", "num_iterations": state['num_iterations'] + 1}

def incorporate_search_results(state: State) -> Dict:
    """Incorporate web search results into the content."""
    messages = state['messages']
    last_message = messages[-1].content
    search_results = state['search_results']
    content = state['content']
    
    if "No web search needed" not in last_message:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant, that has access to internet search results."),
            ("human", f"Refine the following information based on the web search results provided.  Here is the original information/question: {content}. \n\nHere are the web search results: {search_results}")
        ])

        chain = prompt | agent_llm

        response = chain.invoke({})
        new_message = AIMessage(content=response.content)
        return {"messages": messages + [new_message], "content": response.content, "num_iterations": state['num_iterations'], "search_results":""}
    else:
        return {"num_iterations": state['num_iterations']}  # if no search results, nothing to incorporate, so just pass state along

def decide_agent_or_incorporate(state: State) -> str:
    """Decide whether to call the agent again or incorporate information."""
    last_message = state["messages"][-1].content
    if "No web search needed" in last_message:
        return END
    elif state["search_results"]:
        return "incorporate"
    else:
        return "call_agent"
    
def decide_to_continue(state: State) -> str:
    """Decide to stop or continue, based on the iteration count."""
    if state["num_iterations"] > 3:
        return END
    return "call_agent"


# 5. Build graph
graph = StateGraph(State)
graph.add_node("call_agent", call_agent)
graph.add_node("incorporate", incorporate_search_results)

# Define conditional edges
graph.add_conditional_edges("call_agent", decide_agent_or_incorporate, {"incorporate": "incorporate", "call_agent":"call_agent", END: END})
graph.add_conditional_edges("incorporate", decide_to_continue, {"call_agent": "call_agent", END: END})
graph.set_entry_point("call_agent")

runnable = graph.compile()

# 6. Execute with proper input format
inputs = {"messages": [HumanMessage(content="What are the latest news about AI? Summarize them")], "num_iterations": 0, "content":"", "search_results":""}
result = runnable.invoke(inputs)

print("Final content:", result["messages"][-1].content)

# Save the final content as "Untitled.txt" in the current directory.
with open("output.md", "w") as f:
    f.write(result["messages"][-1].content)
