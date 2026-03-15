"""Smoke tests for the compiled graph."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage


def test_graph_importable() -> None:
    """The graph module must be importable without side-effects."""
    from multi_agent_app.graph import graph  # noqa: PLC0415

    assert graph is not None


def test_graph_has_nodes() -> None:
    """The compiled graph must contain the expected node names."""
    from multi_agent_app.graph import graph  # noqa: PLC0415

    node_names = set(graph.nodes.keys())
    assert "agent" in node_names
    assert "tools" in node_names


@pytest.mark.skip(reason="Requires a live OpenAI key – run manually")
def test_graph_invoke_returns_messages() -> None:
    """End-to-end: the graph should return at least one AI message."""
    from langchain_core.messages import AIMessage  # noqa: PLC0415
    from multi_agent_app.graph import graph  # noqa: PLC0415

    result = graph.invoke({"messages": [HumanMessage(content="Hello!")]})
    assert len(result["messages"]) >= 2
    assert isinstance(result["messages"][-1], AIMessage)
