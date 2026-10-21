"""Test cases for the AI Agent Task Automation System."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage  # noqa: E402

from app.agent import OllamaConnectionError  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.tools.calculator import calculator  # noqa: E402
from app.tools.datetime_tool import days_until_date, get_current_datetime  # noqa: E402
from app.tools.file_tool import read_file, summarize_file, write_file  # noqa: E402
from app.tools.search import web_search  # noqa: E402
from app.tools.task_tool import complete_task, create_task, list_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# 1 & 2. Calculator tool (basic + multi-step GST example)
# ---------------------------------------------------------------------------
class TestCalculator:
    def test_basic_arithmetic(self):
        result = calculator.invoke({"expression": "2 + 3 * 4"})
        assert "14" in result

    def test_gst_total_cost_example(self):
        base = calculator.invoke({"expression": "15 * 250"})
        assert "3750" in base
        gst = calculator.invoke({"expression": "3750 * 0.18"})
        assert "675" in gst
        total = calculator.invoke({"expression": "3750 + 675"})
        assert "4425" in total

    def test_invalid_input(self):
        result = calculator.invoke({"expression": "15 items costing each"})
        assert result.startswith("Error")

    def test_division_by_zero(self):
        result = calculator.invoke({"expression": "10 / 0"})
        assert "division by zero" in result.lower()

    def test_empty_expression(self):
        result = calculator.invoke({"expression": "   "})
        assert result.startswith("Error")


# ---------------------------------------------------------------------------
# 3. Date/Time tool
# ---------------------------------------------------------------------------
class TestDateTime:
    def test_get_current_datetime(self):
        result = get_current_datetime.invoke({})
        assert isinstance(result, str)
        assert len(result) > 5

    def test_days_until_date_valid(self):
        result = days_until_date.invoke({"target_date": "2099-12-31"})
        assert "day" in result

    def test_days_until_date_invalid(self):
        result = days_until_date.invoke({"target_date": "not a date"})
        assert result.startswith("Error")


# ---------------------------------------------------------------------------
# 4 & 5. Task tools: creation and listing, completion
# ---------------------------------------------------------------------------
class TestTasks:
    @pytest.fixture(autouse=True)
    def temp_tasks_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TASKS_FILE", str(tmp_path / "tasks.json"))
        yield

    def test_create_and_list_tasks(self):
        created = create_task.invoke({"title": "Prepare ML project"})
        assert "Task created successfully" in created
        listing = list_tasks.invoke({})
        assert "Prepare ML project" in listing
        assert "pending" in listing

    def test_complete_task_by_title(self):
        create_task.invoke({"title": "Prepare ML project"})
        completed = complete_task.invoke({"task": "Prepare ML project"})
        assert "marked as completed" in completed
        listing = list_tasks.invoke({})
        assert "[done]" in listing

    def test_complete_missing_task(self):
        result = complete_task.invoke({"task": "does not exist"})
        assert "no task found" in result.lower()

    def test_malformed_tasks_json(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        monkeypatch.setenv("TASKS_FILE", str(bad))
        listing = list_tasks.invoke({})
        assert listing == "No tasks found."


# ---------------------------------------------------------------------------
# 9. File tools
# ---------------------------------------------------------------------------
class TestFileTools:
    def test_write_and_read_file(self, tmp_path):
        target = tmp_path / "note.txt"
        write_result = write_file.invoke({"path": str(target), "content": "Hello agent world"})
        assert "Successfully wrote" in write_result
        read_result = read_file.invoke({"path": str(target)})
        assert "Hello agent world" in read_result

    def test_read_missing_file(self, tmp_path):
        result = read_file.invoke({"path": str(tmp_path / "missing.txt")})
        assert result.startswith("Error")

    def test_summarize_file(self, tmp_path):
        target = tmp_path / "story.txt"
        target.write_text(
            "The AI agent works hard. It calls tools. The result is summarized.",
            encoding="utf-8",
        )
        result = summarize_file.invoke({"path": str(target)})
        assert "Summary" in result
        assert "Words" in result


# ---------------------------------------------------------------------------
# Search tool error handling
# ---------------------------------------------------------------------------
class TestSearch:
    def test_empty_query(self):
        assert web_search.invoke({"query": "  "}).startswith("Error")

    def test_search_api_failure(self, monkeypatch):
        import duckduckgo_search

        class BrokenDDGS:
            def __init__(self, *args, **kwargs):
                raise ConnectionError("no internet")

        monkeypatch.setattr(duckduckgo_search, "DDGS", BrokenDDGS)
        result = web_search.invoke({"query": "anything"})
        assert result.startswith("Error")


# ---------------------------------------------------------------------------
# Graph routing, tool execution, and conversation memory (fake LLM)
# ---------------------------------------------------------------------------
class _FakeNoToolModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None, **kwargs):
        return AIMessage(content=f"saw {len(messages)} messages")


class _FakeToolModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculator", "args": {"expression": "2 + 3"}, "id": "call_1", "type": "tool_call"}
                ],
            )
        return AIMessage(content="The answer is 5.")


class TestGraph:
    def test_simple_question_no_tools(self):
        graph = build_graph(llm=_FakeNoToolModel())
        out = graph.invoke(
            {"messages": [HumanMessage(content="Hello")]},
            {"configurable": {"thread_id": "t1"}},
        )
        last = out["messages"][-1]
        assert isinstance(last, AIMessage)
        assert last.content.startswith("saw")

    def test_tool_call_execution(self):
        graph = build_graph(llm=_FakeToolModel())
        out = graph.invoke(
            {"messages": [HumanMessage(content="What is 2+3?")]},
            {"configurable": {"thread_id": "t2"}},
        )
        tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "5" in tool_msgs[0].content
        assert out["messages"][-1].content == "The answer is 5."

    def test_conversation_memory(self):
        graph = build_graph(llm=_FakeNoToolModel())
        config = {"configurable": {"thread_id": "memory-thread"}}
        graph.invoke({"messages": [HumanMessage(content="Hi")]}, config)
        out = graph.invoke({"messages": [HumanMessage(content="Again")]}, config)
        # 2nd turn: system + human + ai + human = 4 messages fed to the LLM
        assert out["messages"][-1].content == "saw 4 messages"
        # A different conversation id must not share history
        out2 = graph.invoke(
            {"messages": [HumanMessage(content="Fresh")]},
            {"configurable": {"thread_id": "other-thread"}},
        )
        assert out2["messages"][-1].content == "saw 2 messages"


# ---------------------------------------------------------------------------
# 10. API behavior incl. invalid requests and Ollama unavailable
# ---------------------------------------------------------------------------
class TestAPI:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

        import app.api as api_module

        return TestClient(api_module.app), api_module

    def test_chat_success(self, client, monkeypatch):
        test_client, api_module = client
        monkeypatch.setattr(api_module, "chat", lambda message, conversation_id: f"echo: {message}")
        response = test_client.post("/chat", json={"message": "Create a task"})
        assert response.status_code == 200
        assert response.json()["response"] == "echo: Create a task"

    def test_chat_forwards_conversation_id(self, client, monkeypatch):
        test_client, api_module = client
        seen = {}

        def fake_chat(message, conversation_id):
            seen["cid"] = conversation_id
            return "ok"

        monkeypatch.setattr(api_module, "chat", fake_chat)
        test_client.post("/chat", json={"message": "hi", "conversation_id": "abc"})
        assert seen["cid"] == "abc"

    def test_chat_invalid_request(self, client, monkeypatch):
        test_client, api_module = client

        def fake_chat(message, conversation_id):
            raise ValueError("Message must not be empty.")

        monkeypatch.setattr(api_module, "chat", fake_chat)
        response = test_client.post("/chat", json={"message": ""})
        assert response.status_code == 400

    def test_ollama_unavailable(self, client, monkeypatch):
        test_client, api_module = client

        def fake_chat(message, conversation_id):
            raise OllamaConnectionError("Could not connect to Ollama.")

        monkeypatch.setattr(api_module, "chat", fake_chat)
        response = test_client.post("/chat", json={"message": "hello"})
        assert response.status_code == 503
        assert "Ollama" in response.json()["detail"]

    def test_openai_compatible_endpoint(self, client, monkeypatch):
        test_client, api_module = client
        monkeypatch.setattr(api_module, "chat", lambda message, conversation_id: "done")
        response = test_client.post(
            "/v1/chat/completions",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "done"
        assert body["object"] == "chat.completion"

    def test_openai_models_endpoint(self, client):
        test_client, _ = client
        response = test_client.get("/v1/models")
        assert response.status_code == 200
        assert len(response.json()["data"]) == 1
