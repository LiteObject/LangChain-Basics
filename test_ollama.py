# -------------------- First Example: Basic --------------------
from langchain_ollama import ChatOllama

llm = ChatOllama(model="phi4", temperature=0.5)
# print(llm.invoke("Hello"))

messages = [
    ("system", "You are an experienced stock market analyst."),
    ("user", "What are the top 5 stock symbols?"),
]

# print(llm.invoke(messages))

# -------------------- Second Example: Chaining --------------------
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}")
    ]
)

chain = prompt | llm
# output = chain.invoke({
#     "input": "how are you?",    
#     "input_language": "English", 
#     "output_language": "Bengali"})

# print(output)

# -------------------- Third Example: Tooling --------------------
from typing import List
from langchain_core.tools import tool

@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True

@tool
def greet_user(name: str) -> str:
    """Greet the user by name.
    
    Args:
        name (str): The name of the user.
    """
    return f"Hello, {name}!"

tools = {
    "validate_user": validate_user,
    "greet_user": greet_user,
}

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
).bind_tools([validate_user])

result = llm.invoke(
    "Could you validate user 123? They previously lived at "
    "123 Fake St in Boston MA and 234 Pretend Boulevard in "
    "Houston TX."
)

print("Tool Calls:",result.tool_calls)

# Check if the LLM requested a tool call
if result.tool_calls:
    for tool_call in result.tool_calls:
        # Extract tool name and arguments
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Look up the tool by name in the tools dictionary
        if tool_name in tools:
            tool_to_invoke = tools[tool_name]

            # Use the invoke method to call the tool
            validation_result = tool_to_invoke.invoke(tool_args)

            # Print the result of the tool execution
            print(f"Validation Result for {tool_name}: {validation_result}")
        else:
            print(f"Tool '{tool_name}' not found in the tools dictionary.")
else:
    print("No tool calls were made.")
