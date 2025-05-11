import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph import StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

class AgentState:
    """Represents the state of our agent."""

    messages: list = []
    """The list of messages in the conversation history."""
    next: str = ""
    """The next action or function to execute."""
    repository: str = ""
    """The repository being used."""

def get_github_tools():
    """Create a GitHub Toolkit and get its tools."""

    github = GitHubAPIWrapper()
    toolkit = GitHubToolkit.from_github_api_wrapper(github)
    tools = toolkit.get_tools()
    return tools

def call_llm(state):
    """Calls the LLM to decide the next action."""

    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages(
        [("human", "{messages}")]
    )
    llm = ChatOllama(model="phi4", temperature=0.5)
    chain = prompt | llm
    result = chain.invoke({"messages": messages})
    return {"next": result.content, "messages": messages}

def call_tool(state):
    """Calls the tool decided on by the LLM"""
    messages = state["messages"]
    next = state["next"]
    tools = get_github_tools()
    result = {}

    for tool in tools:
        if tool.name in next:
            result = tool.invoke(next.replace(tool.name, "").strip())
        elif tool.name == "Search Github Issues":
            if "issues" in next.lower():
                result = tool.invoke(next)
        elif tool.name == "List Github Repositories":
            if "repositories" in next.lower() or "repos" in next.lower():
                result = tool.invoke(next)
        elif tool.name == "Get Github File Content":
            if "content" in next.lower():
                result = tool.invoke(next)

    return {"messages": messages + [HumanMessage(content=f"result: {result}")], "next":""}

def decide_next(state):
    """Determines the next step to take in the agent graph."""

    if state["next"] == "" or state["next"] == "call llm":
        return "call_llm"
    return "call_tool"

def build_agent_graph(tools):
    """Builds the agent's LangGraph."""

    graph = StateGraph(AgentState)

    # Define the nodes
    graph.add_node("call_llm", call_llm)
    graph.add_node("call_tool", call_tool)

    # Set the entry point
    graph.set_entry_point("call_llm")

    # Build the edges
    graph.add_conditional_edges(
        "call_llm", decide_next, {
            "call_llm": "call_llm",
            "call_tool": "call_tool",
        }
    )
    graph.add_edge("call_tool", "call_llm")
    graph.add_edge("call_llm", END)

    # Compile and return
    return graph.compile()

if __name__ == "__main__":
    load_dotenv()

    # Set the path to the directory of this file
    base_dir = Path(__file__).parent

    # create the graph
    tools = get_github_tools()
    agent = build_agent_graph(tools)

    # set the env variables
    repo_name = os.getenv("GITHUB_REPOSITORY")
    if not repo_name:
        raise ValueError("GITHUB_REPOSITORY environment variable is not set")

    # Run the agent
    agent.invoke({
        "messages": [HumanMessage(content=f"Overview of existing files in Main branch of {repo_name} repository.")],
        "repository": repo_name,
        "next": "",
    })

