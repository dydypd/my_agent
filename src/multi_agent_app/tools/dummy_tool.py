"""Sample tool – replace with real business-logic tools."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def dummy_tool(query: str) -> str:
    """
    A placeholder tool that echoes the input back to the caller.

    Use this as a template when adding real tools (web search,
    database look-ups, API calls, etc.).

    Parameters
    ----------
    query:
        The input string to echo.

    Returns
    -------
    str
        A confirmation message containing the original query.
    """
    return f"[dummy_tool] received: {query}"
