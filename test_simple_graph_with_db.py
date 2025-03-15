from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Define the State
class State(TypedDict):
    count: int

# Create the graph
graphBuilder = StateGraph(State)

# Define the nodes
def printer(state: State) -> State:
    """Prints the state to the console"""
    print(f"Count: {state['count']}")
    state['count'] += 1 # Increment the count
    return state # Return the updated state

# Add the nodes to the graph
graphBuilder.add_node("printer", printer)

# Set the entry point
# graphBuilder.set_entry_point("printer")
graphBuilder.add_edge(START, "printer")
graphBuilder.add_edge("printer", END)

# Configuration and memory
config = {"configurable": {"thread_id": 1}}
conn = sqlite3.connect('checkpoints.sqlite', check_same_thread=False)
memory = SqliteSaver(conn)

# Compile the graph
graph = graphBuilder.compile(checkpointer=memory)

# Check if a saved state exists
current_state = graph.get_state(config)
if current_state is None:
    # First run: use the initial state
    inputs = {"count": 0}  # Initial state
    result = graph.invoke(inputs, config)
else:
    # Subsequent runs: use the saved state
    result = graph.invoke(current_state.values, config)
print(result)  # Output the result

# Draw the graph
try:
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path="diagram_simple_graph.png")
except Exception as e:
    print(e)

# Run the bot
while True:
    user_input = input(">> ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting...")
        break

    count = graph.get_state(config).values["count"]
    result = graph.invoke({"count": count}, config)
    print(result) 