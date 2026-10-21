"""Web search tool backed by DuckDuckGo (no API key required)."""

import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query, e.g. 'latest AI news 2026'.")


@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """Search the web and return a summary of the top results."""
    logger.info("web_search input: %s", query)
    if not query or not query.strip():
        return "Error: query must not be empty."
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: duckduckgo-search package is not installed (pip install duckduckgo-search)."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results found for '{query}'."
        lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"- {title}: {body} ({href})")
        logger.info("web_search returned %d results", len(results))
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - network/library failures
        logger.error("web_search failed: %s", exc)
        return f"Error: web search failed for '{query}': {exc}"
