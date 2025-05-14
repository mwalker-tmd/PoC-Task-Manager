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
