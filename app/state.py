"""Shared state definition for the LangGraph agent."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State passed between graph nodes.

    ``add_messages`` ensures new messages are appended to the conversation
    history instead of replacing it, which gives us memory across turns
    when a checkpointer is used.
    """

    messages: Annotated[list[BaseMessage], add_messages]
