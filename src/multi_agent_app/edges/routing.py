"""Conditional routing logic for the agent graph."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage

from ..state import AgentState


def _has_pending_tool_calls(state: AgentState) -> bool:
    """Return True when the last model message requests one or more tools."""
    if not state.get("messages"):
        return False

    last_message: BaseMessage = state["messages"][-1]
    return bool(getattr(last_message, "tool_calls", None))


def should_continue_profile(state: AgentState) -> Literal["profile_tools", "end"]:
    """Route profile node to its own tool node only."""
    if _has_pending_tool_calls(state):
        return "profile_tools"
    return "end"


def should_continue_contact(state: AgentState) -> Literal["contact_tools", "end"]:
    """Route contact node to its own tool node only."""
    if _has_pending_tool_calls(state):
        return "contact_tools"
    return "end"


def route_next(state: AgentState) -> Literal["profile_node", "contact_node", "chatter_node", "end"]:
    """
    Route from supervisor to the selected sub-agent.

    Supervisor writes its choice to ``state['next']``.
    """
    decision = str(state.get("next") or "")
    if decision == "profile_node":
        return "profile_node"
    if decision == "contact_node":
        return "contact_node"
    if decision == "chatter_node":
        return "chatter_node"
    return "end"
