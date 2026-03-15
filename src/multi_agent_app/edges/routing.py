"""Conditional routing logic for the agent graph."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage

from ..state import AgentState


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Decide whether the graph should invoke a tool or finish.

    If the last message produced by the model contains tool-call
    requests, route to the ``"tools"`` node; otherwise finish.

    Parameters
    ----------
    state:
        The current :class:`~multi_agent_app.state.AgentState`.

    Returns
    -------
    Literal["tools", "end"]
        ``"tools"`` when the model requested a tool call,
        ``"end"``   when the model produced a final answer.
    """
    last_message: BaseMessage = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"
