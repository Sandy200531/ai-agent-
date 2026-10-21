"""Agent factory and high-level chat interface with conversation memory."""

import logging

from langchain_core.messages import AIMessage

from app.graph import build_graph

logger = logging.getLogger(__name__)

_graph = None


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


def get_graph():
    """Return a cached compiled graph (keeps in-memory conversation state)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _is_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        key in text
        for key in ("connection refused", "connecterror", "failed to connect", "errno 111", "10061", "cannot connect", "name resolution")
    )


def chat(message: str, conversation_id: str = "default") -> str:
    """Send a user message to the agent and return the final response text."""
    if not message or not message.strip():
        raise ValueError("Message must not be empty.")

    logger.info("User request: %s", message)
    graph = get_graph()
    config = {"configurable": {"thread_id": conversation_id}}

    try:
        result = graph.invoke({"messages": [("human", message)]}, config)
    except Exception as exc:  # noqa: BLE001
        if _is_connection_error(exc):
            logger.error("Ollama connection error: %s", exc)
            raise OllamaConnectionError(
                "Could not connect to Ollama. Please make sure Ollama is running "
                "(run 'ollama serve' and pull the configured model)."
            ) from exc
        logger.exception("Agent failed to process the request")
        raise

    messages = result.get("messages", [])
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and last.content:
        response = last.content if isinstance(last.content, str) else str(last.content)
    else:
        response = "I could not produce a response. Please try again."
    logger.info("Final response: %s", response[:500])
    return response
