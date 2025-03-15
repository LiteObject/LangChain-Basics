import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentType, initialize_agent
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

# Set your environment variables using os.environ
# os.environ["GITHUB_APP_ID"] = "1171958"
# os.environ["GITHUB_APP_PRIVATE_KEY"] = "./my-langchain-app.2025-03-09.private-key.pem"
# os.environ["GITHUB_REPOSITORY"] = "LiteObject/ProductManagement"

# Optional
# os.environ["GITHUB_BRANCH"] = "main"
# os.environ["GITHUB_BASE_BRANCH"] = "main"

llm = ChatOllama(model="phi4", temperature=0.5)
github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)
tools = toolkit.get_tools()

# STRUCTURED_CHAT includes args_schema for each tool, helps tool args parsing errors.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# print("Available tools:")
# for tool in tools:
#     print("\t" + tool.name)

# Read repo name from environment variable
repo_name = os.getenv("GITHUB_REPOSITORY")
if not repo_name:
    raise ValueError("GITHUB_REPOSITORY environment variable is not set")

try:
    agent.invoke(f"Overview of existing files in Main branch of '{repo_name}' repository.")
    # agent.invoke("List all open issues in the LiteObject/ProductManagement repository.")
    # agent.invoke(
    #     f"You are a Google Principal software engineer tasked with resolving bug fix issues in the {repo_name} repository.",
    #     "Your tasks are:"
    #     "1. Examine the current open issues."
    #     "2. Determine which issues are labeled as 'bug'."
    #     "3. For each 'bug' issue, create a code change that resolve the issue."
    #     "4. Add comments on the issue with the code changes."
    #     "5. For this run only work on a single issue."
    #     "6. If any part of this fails log the error and exit."
    # )
except Exception as e:
    print(e)
