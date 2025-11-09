# GenAI Product Recommendation Engine

## Overview
This project implements a Retrieval-Augmented Generation (RAG) engine for product recommendation using Model Context Protocol (MCP). It leverages LangGraph for agent orchestration and integrates various modules for perception, decision-making, memory, and action.

## Tech Stack
- **Python**
- **LangGraph**: Agent workflow orchestration
- **Model Context Protocol (MCP)**: Standardized context management
- **Custom Modules**: Perception, Decision, Memory, Action

## Workflow: How a Query is Processed
1. **Perception**: The user's query is received and analyzed for intent and context (`perception.py`).
2. **Memory**: Relevant historical data and context are retrieved (`memory.py`).
3. **Decision**: The system determines the best course of action based on the query and context (`decision.py`).
4. **Action**: Executes the chosen action, such as retrieving product data or generating recommendations (`action.py`).
5. **Graph Orchestration**: LangGraph manages the flow between modules, ensuring the correct sequence and context passing (`graph.py`, `agent_langgraph.py`).
6. **MCP Server**: Handles external requests and serves results (`mcp_server.py`).

## File Structure
- `action.py`: Executes actions based on decisions.
- `agent_langgraph.py`: LangGraph agent orchestration logic.
- `decision.py`: Decision-making logic.
- `graph.py`: Workflow graph definition.
- `json_analyzer.py`: JSON data analysis utilities.
- `log_utils.py`: Logging utilities.
- `mcp_server.py`: MCP server implementation.
- `memory.py`: Context and memory management.
- `models.py`: Data models and schemas.
- `nodes.py`: Workflow nodes for LangGraph.
- `perception.py`: Query analysis and perception logic.
- `state.py`: State management.

## Getting Started
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the agent:
   ```bash
   python agent_langgraph.py
   ```

## Example Query Flow
1. User sends a product-related query to the agent.
2. The query is processed by the perception module.
3. Memory module retrieves relevant context.
4. Decision module selects the best recommendation strategy.
5. Action module fetches/generates recommendations.
6. Response is returned to the user.
