"""Shared state definitions for the agent graph."""

from __future__ import annotations

import operator
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    The canonical state object passed between every node in the graph.

    Fields
    ------
    messages:
        The conversation history.  Using ``operator.add`` as the reducer
        means each node *appends* its new messages rather than replacing
        the full list – this is the standard LangGraph pattern.
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
