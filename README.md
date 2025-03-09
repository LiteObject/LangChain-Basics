# What is LangChain 🦜️🔗

LangChain is an open-source library that helps developers build applications with large language models (LLMs). It provides a standard interface for chains, which combine LLMs with other computation or knowledge sources. This makes it easier to develop LLM-powered applications.

This framework consists of several parts.

- **LangChain Libraries**: The Python and JavaScript libraries. Contains interfaces and integrations for a myriad of components, a basic run time for combining these components into chains and agents, and off-the-shelf implementations of chains and agents.

- **LangChain Templates**: A collection of easily deployable reference architectures for a wide variety of tasks. (Python only)

- **LangServe**: A library for deploying LangChain chains as a REST API. (Python only)

- **LangSmith**: A developer platform that lets you debug, test, evaluate, and monitor chains built on any LLM framework and seamlessly integrates with LangChain.

## LangChain Architecture
![LangChain Stack](./res/langchain_stack_dark.svg)

### langchain-core
Base abstractions and LangChain Expression Language.

### langchain-community
Third party integrations.

### langchain
Chains, agents, and retrieval strategies that make up an application's cognitive architecture.

### langgraph: 
Build robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.

### langserve: 
Deploy LangChain chains as REST APIs.

### LangSmith: 
A developer platform that lets you debug, test, evaluate, and monitor LLM applications.

## LangChain Expression Language (LCEL)
[LCEL](https://js.langchain.com/v0.1/docs/get_started/introduction/#langchain-expression-language-lcel) is a declarative way to compose chains.


---
## Links
- [LangChain Introduction](https://python.langchain.com/docs/get_started/introduction)
- [GitHub: langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch?tab=readme-ov-file)
