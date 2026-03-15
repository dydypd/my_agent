#!/usr/bin/env bash
# =============================================================================
# setup.sh – Scaffold the Enterprise-Grade LangGraph Multi-Agent boilerplate.
# Usage: bash setup.sh [project-name]
# Default project name: my_enterprise_agent
# =============================================================================

set -euo pipefail

PROJECT_NAME="${1:-my_enterprise_agent}"

echo "==> Creating project: $PROJECT_NAME"

# ---------------------------------------------------------------------------
# 1. Top-level directories
# ---------------------------------------------------------------------------
mkdir -p "$PROJECT_NAME"/src/multi_agent_app/tools
mkdir -p "$PROJECT_NAME"/src/multi_agent_app/nodes
mkdir -p "$PROJECT_NAME"/src/multi_agent_app/edges
mkdir -p "$PROJECT_NAME"/tests

# ---------------------------------------------------------------------------
# 2. Top-level config files
# ---------------------------------------------------------------------------

cat > "$PROJECT_NAME"/.env.example <<'EOF'
# ---- OpenAI ----
OPENAI_API_KEY=sk-...

# ---- LangSmith (optional tracing) ----
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my_enterprise_agent
EOF

cat > "$PROJECT_NAME"/langgraph.json <<'EOF'
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/multi_agent_app/graph.py:graph"
  },
  "env": ".env"
}
EOF

cat > "$PROJECT_NAME"/pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "multi_agent_app"
version = "0.1.0"
description = "Enterprise-grade LangGraph Multi-Agent boilerplate"
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3",
    "langchain-openai>=0.2",
    "langgraph>=0.2",
    "langgraph-cli[inmem]>=0.1.55",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.10",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
EOF

# ---------------------------------------------------------------------------
# 3. Package init files
# ---------------------------------------------------------------------------
touch "$PROJECT_NAME"/src/multi_agent_app/__init__.py
touch "$PROJECT_NAME"/src/multi_agent_app/tools/__init__.py
touch "$PROJECT_NAME"/src/multi_agent_app/nodes/__init__.py
touch "$PROJECT_NAME"/src/multi_agent_app/edges/__init__.py
touch "$PROJECT_NAME"/tests/__init__.py

# ---------------------------------------------------------------------------
# 4. Source files (written inline for self-contained bootstrap)
# ---------------------------------------------------------------------------

cat > "$PROJECT_NAME"/src/multi_agent_app/configuration.py <<'EOF'
"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Configuration:
    """Central configuration object for the agent application."""

    # LLM settings
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096

    # LangSmith / tracing
    langchain_project: str = "multi_agent_app"

    @classmethod
    def from_env(cls) -> "Configuration":
        """Construct a Configuration instance from the current environment."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", "0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            langchain_project=os.getenv("LANGCHAIN_PROJECT", "multi_agent_app"),
        )
EOF

cat > "$PROJECT_NAME"/src/multi_agent_app/state.py <<'EOF'
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
EOF

cat > "$PROJECT_NAME"/src/multi_agent_app/models.py <<'EOF'
"""LLM factory helpers."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .configuration import Configuration


def get_llm(config: Configuration | None = None) -> ChatOpenAI:
    """
    Return an initialised ChatOpenAI instance.

    Parameters
    ----------
    config:
        Optional :class:`Configuration` object.  When *None* a default
        instance is created from the current environment.
    """
    cfg = config or Configuration.from_env()
    return ChatOpenAI(
        model=cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
EOF

cat > "$PROJECT_NAME"/src/multi_agent_app/tools/dummy_tool.py <<'EOF'
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
EOF

cat > "$PROJECT_NAME"/src/multi_agent_app/nodes/agent_node.py <<'EOF'
"""Core agent node – binds the LLM to available tools and invokes it."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..models import get_llm
from ..state import AgentState
from ..tools.dummy_tool import dummy_tool

# All tools available to the agent
TOOLS = [dummy_tool]


def agent_node(state: AgentState) -> AgentState:
    """
    Primary agent node.

    1. Binds the configured LLM to the registered tools.
    2. Invokes the model with the current message history.
    3. Returns a *partial* state update containing only the new message
       (LangGraph merges it with the existing state via ``operator.add``).

    Parameters
    ----------
    state:
        The current :class:`~multi_agent_app.state.AgentState`.

    Returns
    -------
    AgentState
        A partial state dict with the model's response appended to
        ``messages``.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    response: AIMessage = llm_with_tools.invoke(state["messages"])

    # Return only the delta; the ``operator.add`` reducer handles merging.
    return {"messages": [response]}
EOF

cat > "$PROJECT_NAME"/src/multi_agent_app/edges/routing.py <<'EOF'
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
EOF

cat > "$PROJECT_NAME"/src/multi_agent_app/graph.py <<'EOF'
"""
Assemble and compile the LangGraph StateGraph.

The ``graph`` symbol exported here is what LangGraph Studio and
``langgraph dev`` will discover via ``langgraph.json``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .edges.routing import should_continue
from .nodes.agent_node import TOOLS, agent_node
from .state import AgentState

# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)

# Nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(TOOLS))

# Edges
builder.add_edge(START, "agent")

# Conditional edge: stay in the loop while the model calls tools;
# exit when the model produces a final answer.
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

# After every tool execution, hand control back to the agent.
builder.add_edge("tools", "agent")

# ---------------------------------------------------------------------------
# Compile – this is what LangGraph Studio imports.
# ---------------------------------------------------------------------------
graph = builder.compile()

# Human-readable name visible in LangGraph Studio
graph.name = "Multi-Agent App"
EOF

cat > "$PROJECT_NAME"/tests/test_graph.py <<'EOF'
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
EOF

echo ""
echo "==> Project '$PROJECT_NAME' created successfully."
echo "    Next steps:"
echo "      cd $PROJECT_NAME"
echo "      cp .env.example .env   # fill in your API keys"
echo "      pip install -e '.[dev]'"
echo "      langgraph dev          # launch LangGraph Studio"
