# 🤖 AGENTIC RAG Codebase for Product Recommendation

A production-ready, **Hexagonal Architecture** (Ports and Adapters) implementation of an Agentic AI system tailored for E-commerce Product Discovery. This system leverages **LangGraph** for orchestration, **Google Gemini** for reasoning, and the **Model Context Protocol (MCP)** for standardized tool execution.

---

## 🏗️ Architecture: Hexagonal & Domain-Driven

This project strictly follows **Hexagonal Architecture** instructions to decouple the core business logic from external tools and frameworks. This ensures testability, maintainability, and the ability to swap infrastructure components (like LLMs or Vector DBs) without touching the core agent logic.

### The Layers

1.  **🟢 Domain Layer (`src/domain`)**
    *   **The Core**: Contains pure business logic, data models, and interface definitions (Ports).
    *   **No Dependencies**: This layer *never* imports external infrastructure libraries (like `google.genai` or `faiss`). It only uses Python standard libraries and Pydantic.
    *   **Subdomains**: Structured by bounded contexts:
        *   `perception`: Logic for understanding user intent.
        *   `decision`: Logic for planning and reasoning ("Should I search or answer?").
        *   `memory`: Logic definitions for conversation history.
        *   `tools`: Definitions of what tools look like.

2.  **🟡 Application Layer (`src/application`)**
    *   **The Orchestrator**: Wires the Domain Logic together to perform Use Cases.
    *   **Services**: `PerceptionService` and `DecisionService` implementation reside here. They use the *Ports* defined in the Domain to execute tasks.
    *   **Agent Workflow**: Defines the LangGraph state machine (`agent_orchestrator.py`) that manages the cognitive cycle (Perceive → Remember → Decide → Act).

3.  **🔴 Infrastructure Layer (`src/infrastructure`)**
    *   **The Adapters**: Technical implementations of the Domain Ports.
    *   **Gemini Adapter**: Takes the generic `LLMProvider` port and implements it using Google's GenAI SDK.
    *   **FAISS Adapter**: Implements `MemoryStore` using local vector indices.
    *   **MCP Adapter**: Implements `ToolExecutor` by connecting to an external MCP server process.

4.  **🔵 Composition Root (`main.py`)**
    *   The only entry point that knows about all layers.
    *   Performs **Dependency Injection**: It instantiates the Adapters (Infrastructure) and injects them into the Services (Application).

---

## 📂 Project Structure

```text
RAG-MCP/
├── main.py                     # Entry point (Dependency Injection & Startup)
├── server/
│   └── mcp_server.py           # Standalone MCP Server (Tools & RAG Engine)
├── src/
│   ├── domain/                 # CORE: Models & Ports (Interfaces)
│   │   ├── models/             # Shared state & schemas
│   │   ├── ports/              # Output Interfaces (LLM, Memory, Tools)
│   │   ├── perception/         # Perception domain models
│   │   ├── decision/           # Decision domain models
│   │   └── memory/             # Memory domain models
│   ├── application/            # ORCHESTRATION: Business Logic
│   │   ├── services/           # Implementation of services (Agent Logic)
│   │   └── ports/              # Input Ports (Use Cases)
│   └── infrastructure/         # TOOLS: Adapters & Config
│       └── adapters/
│           └── output/         # Adapters for Gemini, FAISS, MCP
└── documents/                  # Raw product data (JSON)
```

---

## 🚀 Key Features

*   **Cognitive Cycle**: Implements a `Perceive -> Remember -> Decide -> Act` loop using LangGraph.
*   **Model Context Protocol (MCP)**: All product search and retrieval tools are hosted on a separate MCP server, compliant with the Anthropic MCP standard.
*   **Structured Reasoning**: Uses Pydantic models to enforce structured output from the LLM, reducing hallucinations.
*   **Swappable Components**: Want to use OpenAI instead of Gemini? Just create `OpenAIAdapter` in infrastructure and swap it in `main.py`.

---

## 🛠️ Setup & Usage

### Prerequisites
*   Python 3.10+
*   Google Gemini API Key

### Installation

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```
3.  **Environment Setup**:
    Create a `.env` file in the root directory:
    ```ini
    GEMINI_API_KEY=your_key_here
    ```

### Running the Agent

Start the agent CLI. This will automatically spark up the MCP server in a subprocess.

```bash
python main.py "Find me some red nike running shoes under $100"
```

### Running the MCP Server (Dev Mode)

If you want to debug the tools or vector search independently:

```bash
python server/mcp_server.py dev
```

---

## 🧠 Cognitive Flow (Under the Hood)

1.  **User Input**: "I want a hiking bag."
2.  **Perception Service**: Analyzes intent (`ProductSearch`) and entities (`Category: Bag`, `Activity: Hiking`).
3.  **Memory Service**: Retrives past preferences (e.g., "User prefers waterproof gear").
4.  **Decision Service**: Generates a plan.
    *   *Thought*: "I need to search for hiking bags with water resistance."
    *   *Action*: `CallTool: search_product_documents(query="hiking bag waterproof")`
5.  **Tool Execution (MCP)**: The generic `MCPToolAdapter` sends the request to `mcp_server.py`, which consults the FAISS index.
6.  **Loop**: The agent receives results, refines them, and eventually produces a `FINAL_ANSWER`.
