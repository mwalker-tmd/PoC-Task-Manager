# Task Manager PoC with LangGraph

A proof of concept task manager built using LangGraph and LangChain.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Poetry for dependency management
- API keys for:
  - OpenAI
  - Anthropic
  - Tavily

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   TAVILY_API_KEY=your_tavily_key
   ```

### Running the Server

To start the LangGraph server and launch LangGraph Studio:

```bash
langgraph dev
```

This will:
1. Start the LangGraph server
2. Open LangGraph Studio in your default web browser
3. Allow you to interact with your agent through the Studio interface

The server will be available at http://127.0.0.1:2024

## Running the LangGraph Dev Server

To use LangGraph Studio with this project:

1. **Install dependencies** (if you haven't already):
   ```sh
   uv pip install -e .
   ```

2. **Start the LangGraph dev server:**
   ```sh
   langgraph dev
   ```
   - This will read your `langgraph.json` and expose your agent graphs to LangGraph Studio.
   - You can then open [http://localhost:2024](http://localhost:2024) (or the port shown in your terminal) to access LangGraph Studio.

>Next up will be getting the FastMCP Inspector connecting

## Notes

- The OpenAPI docs for your FastAPI endpoints are available at [http://localhost:8000/docs](http://localhost:8000/docs) if you run:
  ```sh
  uvicorn backend.mcp_server:app
  ```
- The LangGraph Studio session will only work with the `langgraph dev` server, not with FastAPI/FastMCP.

## Work in Progress

This is a WIP demo. The integration between FastAPI, FastMCP, and LangGraph is evolving. You will see some example code still in some files. See comments in the code for more details.
