"""LangGraph workflow definition for the agent, including memory support."""

import logging
from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import get_settings
from app.state import AgentState
from app.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful task-automation assistant. You have access to tools for "
    "calculations, web search, date/time, task management, and file operations. "
    "Use tools whenever they help answer the user accurately. After using tools, "
    "summarize the results clearly for the user."
)


def _build_llm():
    from langchain_ollama import ChatOllama

    settings = get_settings()
    logger.info(
        "Using Ollama model '%s' at %s", settings.ollama_model, settings.ollama_base_url
    )
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def build_graph(llm=None):
    """Build and compile the LangGraph agent graph.

    Flow: START -> agent -> (tool calls? -> tool_executor -> agent)* -> final_response -> END
    """
    if llm is None:
        llm = _build_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm_with_tools.invoke(messages)
        logger.info(
            "Agent decision: %s",
            "tool_call" if getattr(response, "tool_calls", None) else "direct answer",
        )
        return {"messages": [response]}

    tool_executor = ToolNode(ALL_TOOLS)

    def tool_executor_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        for tc in getattr(last, "tool_calls", []) or []:
            logger.info("Tool input for '%s': %s", tc["name"], tc.get("args"))
        result = tool_executor.invoke(state)
        for msg in result.get("messages", []):
            logger.info("Tool output from '%s': %s", msg.name, str(msg.content)[:500])
        return result

    def final_response_node(state: AgentState) -> dict:
        logger.info("Final response ready")
        return {}

    def route_after_agent(state: AgentState) -> Literal["tool_executor", "final_response"]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                logger.info("Agent selected tool '%s' with args %s", tc["name"], tc.get("args"))
            return "tool_executor"
        return "final_response"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("final_response", final_response_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tool_executor", "agent")
    graph.add_edge("final_response", END)

    return graph.compile(checkpointer=MemorySaver())
