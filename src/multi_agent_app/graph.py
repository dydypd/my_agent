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
