# LangChain: Building Powerful Applications with Large Language Models (LLMs) 🦜️🔗

LangChain is an open-source framework designed to simplify the creation of applications powered by Large Language Models (LLMs). It provides a robust set of tools, components, and interfaces that allow developers to seamlessly integrate LLMs with other computational resources and knowledge sources. This enables the development of sophisticated, context-aware applications that go beyond the capabilities of standalone LLMs.

## Key Features and Components

LangChain's ecosystem is built around several core components:

-   **LangChain Libraries (Python & JavaScript):**
    -   **Interfaces and Integrations:** A vast collection of pre-built integrations with various LLMs, vector databases, document loaders, and other tools.
    -   **Chains and Agents:** A flexible runtime for composing LLM-powered workflows (chains) and autonomous agents that can interact with the environment.
    -   **Off-the-Shelf Implementations:** Ready-to-use chains and agents for common tasks, accelerating development.

-   **LangChain Templates (Python):**
    -   **Reference Architectures:** A curated collection of deployable application blueprints for a wide range of use cases, providing a solid foundation for your projects.

-   **LangServe (Python):**
    -   **API Deployment:** A streamlined solution for deploying LangChain chains as RESTful APIs, making it easy to integrate your LLM applications into other systems.

-   **LangSmith:**
    -   **Development Platform:** A comprehensive platform for debugging, testing, evaluating, and monitoring LLM applications built with any framework.
    -   **Seamless LangChain Integration:** Deep integration with LangChain for enhanced observability and control.

-   **LangGraph:**
    - **Stateful Multi-Actor Applications:** Build complex applications with multiple actors and state management.
    - **Graph-Based Architecture:** Model steps as nodes and edges in a graph, enabling robust and flexible workflows.

## LangChain Architecture: A Layered Approach

LangChain's architecture is designed for modularity and extensibility. It can be visualized as a stack:

![LangChain Stack](./res/langchain_stack_dark.svg)

-   **langchain-core:**
    -   **Foundation:** Provides the fundamental abstractions and the LangChain Expression Language (LCEL).
    -   **LCEL:** A declarative language for composing chains, enabling efficient and expressive workflow definitions.

-   **langchain-community:**
    -   **Integrations Hub:** A growing collection of third-party integrations, expanding the capabilities of LangChain.

-   **langchain:**
    -   **Application Logic:** Contains the core building blocks for creating LLM applications, including chains, agents, and retrieval strategies.

- **langgraph:**
    - **Graph-Based Workflows:** Build robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.

-   **langserve:**
    -   **Deployment:** Enables the deployment of LangChain chains as REST APIs.

-   **LangSmith:**
    -   **Observability:** A developer platform for debugging, testing, evaluating, and monitoring LLM applications.

## LangChain Expression Language (LCEL)

[LCEL](https://js.langchain.com/v0.1/docs/get_started/introduction/#langchain-expression-language-lcel) is a powerful declarative way to compose chains. It allows you to define complex workflows in a concise and readable manner.

## Getting Started

Ready to start building with LangChain? Here are some helpful resources:

-   **LangChain Introduction:** [https://python.langchain.com/docs/get_started/introduction](https://python.langchain.com/docs/get_started/introduction)
-   **LangChain Python Documentation:** [https://python.langchain.com/docs/](https://python.langchain.com/docs/)
-   **LangChain JavaScript Documentation:** [https://js.langchain.com/docs/](https://js.langchain.com/docs/)
-   **GitHub: langchain-ai/langchain:** [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
-   **GitHub: langchain-ai/rag-from-scratch:** [https://github.com/langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)

