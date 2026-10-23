# AI Agent Task Automation System

A multi-step AI agent built with **Python, LangGraph, Ollama, and Open WebUI**.
It understands user requests, selects tools, executes them step by step,
observes results, and returns a concise final answer — without exposing the
model's internal chain-of-thought.

## Features

- 🧠 **LangGraph agent workflow** with `agent`, `tool_executor`, and `final_response` nodes
- 🔀 **Conditional routing** — the agent loops through tools until the task is done
- 🧰 **5 tool types**: calculator, web search, date/time, task manager, file tools
- 💬 **Conversation memory** per `conversation_id` (in-memory checkpointer)
- 🌐 **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/models`) for easy Open WebUI integration
- 🛡️ **Graceful error handling** for invalid input, missing files, bad JSON, API/Ollama failures
- 📋 **Structured logging** of requests, decisions, tool calls, and responses (no secrets logged)
- ✅ **Pytest suite** covering tools, graph routing, memory, and API behavior

## Architecture

```
User Input
    ↓
┌──────────┐         tool_calls?        ┌────────────────┐
│  agent   │ ───────── yes ───────────▶ │ tool_executor  │
│  (LLM)   │ ◀───────────────────────── │ (run tools)    │
└──────────┘         observations       └────────────────┘
    │ no tool calls
    ↓
┌────────────────┐
│ final_response │ ──▶ User
└────────────────┘
```

## Project Structure

```text
ai-agent/
│
├── app/
│   ├── __init__.py
│   ├── agent.py            # Agent entry point + conversation memory
│   ├── graph.py            # LangGraph workflow definition
│   ├── state.py            # Agent state schema
│   ├── config.py           # Environment-based configuration
│   ├── api.py              # FastAPI app (OpenAI-compatible)
│   └── tools/
│       ├── __init__.py
│       ├── calculator.py
│       ├── search.py
│       ├── datetime_tool.py
│       ├── task_tool.py
│       └── file_tool.py
│
├── data/
│   └── tasks.json
│
├── tests/
│   └── test_agent.py
│
├── conftest.py
├── run.py                  # Starts the API server
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Requirements

- Python **3.11 or 3.12+** (developed on 3.13)
- [Ollama](https://ollama.com) installed and running

### 2. Install dependencies

```bash
cd ai-agent
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env
```

`.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
HOST=0.0.0.0
PORT=8000
```

> Use a tool-calling-capable model. `llama3.2`, `llama3.1`, or `qwen2.5` work well
> with Ollama tool calling.

### 4. Install and start Ollama (Windows)

```powershell
# Install from https://ollama.com/download (or via winget)
winget install Ollama.Ollama

# Start the server (usually already running after install)
ollama serve

# Pull the configured model
ollama pull llama3.2
```

Verify:

```bash
ollama list
curl http://localhost:11434/api/tags
```

### 5. Run the agent API

```bash
python run.py
```

The API starts at `http://localhost:8000` (`POST /chat`, OpenAI-compatible
`/v1/*` endpoints, and `GET /health`).

## Using the API

Simple chat:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a task to finish my ML project", "conversation_id": "demo"}'
```

```json
{"response": "Task created successfully: 'finish my ML project' (id: ...)."}
```

OpenAI-compatible request (used by Open WebUI):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "What is 15*250*1.18?"}]
  }'
```

## Open WebUI Configuration

1. Start the agent: `python run.py`
2. Open WebUI → **Settings → Connections**
3. Add an **OpenAI-compatible** connection:
   - **URL**: `http://localhost:8000/v1`
   - **API key**: any non-empty value (e.g. `dummy`) — the local server ignores it
4. Click **Save**, then pick the model listed under `/v1/models`
   (the value of `OLLAMA_MODEL` from `.env`) in the chat model selector.
5. Chat as usual — each Open WebUI user/session maps to a `conversation_id`, so
   context like "mark **it** as completed" works within the same conversation.

> Tip: Open WebUI sends a `user` field; the API uses it (falling back to
> `conversation_id`, then `default`) to isolate conversation memory.

## Tools

| Tool | Description |
|------|-------------|
| `calculator` | Safe arithmetic evaluation (`sqrt`, `abs`, `log`, `pi`, …) |
| `web_search` | DuckDuckGo search, returns top 5 results |
| `get_current_datetime` | Current local date/time |
| `days_until_date` | Days remaining until a target date |
| `create_task` / `list_tasks` / `complete_task` | JSON-backed to-do list (`data/tasks.json`) |
| `read_file` / `write_file` / `summarize_file` | Text file read, write, and extractive summary |

## Example Conversations

**Calculator + GST (multi-tool reasoning)**

```
You:   Calculate the total cost of 15 items costing ₹250 each and add 18% GST.
Agent: (calls calculator 3 times: 15*250 → 3750; 3750*0.18 → 675; 3750+675 → 4425)
Agent: The base cost is ₹3,750, GST (18%) is ₹675, so the final total is ₹4,425.
```

**Task memory in one conversation (`conversation_id: "demo"`)**

```
You:   Create a task called Prepare ML project.
Agent: Task created successfully: 'Prepare ML project' (id: a1b2c3d4).

You:   Mark it as completed.
Agent: Task 'Prepare ML project' marked as completed.
```

**Full multi-step example**

```
You: Create a task called 'Complete AI project', calculate how many days are
     left until December 31, and save the result to a text file.

Agent: (1) calls create_task  (2) calls days_until_date  (3) calls write_file
Agent: ✅ Task 'Complete AI project' created.
       ⏳ Days left until 2026-12-31: 87.
       📝 Saved the result to ai_project.txt.
```

## Testing

```bash
python -m pytest tests/test_agent.py -v
```

Covers:

1. Simple question without tools
2. Calculator tool (basic + GST multi-step + invalid input)
3. Task creation
4. Task listing
5. Multi-step task / tool chain routing through the graph
6. Multiple tool calls
7. Invalid tool input handling
8. Ollama unavailable (503 + friendly message)
9. File reading
10. Conversation context / memory across turns

> The graph tests use a fake LLM, so **no Ollama server is required** to run the
> test suite.

## Error Handling

| Failure | Behavior |
|---------|----------|
| Invalid tool input (bad math, bad date) | Tool returns an `Error: ...` string the LLM can reason about |
| Missing file / malformed JSON | Per-tool error messages; tasks file degrades to empty list |
| Ollama not running | `503` with "Could not connect to Ollama…" |
| Empty user message | `400` with validation message |
| Tool execution failure | `ToolNode` surfaces an error `ToolMessage` to the agent |

## Resume-Ready Description

> **AI Agent Task Automation System** — Python, LangGraph, Ollama, Open WebUI
>
> - Designed and built a multi-step ReAct-style LLM agent with LangGraph, using
>   conditional edges and a tool loop (`agent → tool_executor → agent → final_response`).
> - Implemented typed, Pydantic-validated tools (calculator, web search, datetime,
>   JSON-backed task manager, file I/O) with robust error handling and logging.
> - Exposed the agent through a FastAPI server with an OpenAI-compatible
>   `/v1/chat/completions` endpoint for native Open WebUI integration, plus
>   per-conversation memory via LangGraph checkpointers.
> - Added a pytest suite (26 tests) covering tools, graph routing, memory, API
>   contracts, and failure modes; runs fully offline with a fake LLM.

## Legacy Folder

An earlier experimental scaffold lives in `langgraph_agent/` (`main.py` CLI).
It is **not** part of the spec-compliant `app/` implementation and can be
removed with:

```powershell
Remove-Item -Recurse langgraph_agent, main.py
```
## Quick Run 

first terminal activate venv ,start ollama serve(ollama server),python run.py  and in another terminal start open-webui serve(openweb-ui server), now open browser http://localhost:8080/ this link 
