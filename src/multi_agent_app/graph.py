"""
Assemble and compile the LangGraph StateGraph.

The ``graph`` symbol exported here is what LangGraph Studio and
``langgraph dev`` will discover via ``langgraph.json``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from multi_agent_app.edges.routing import (
    route_next,
    should_continue_contact,
    should_continue_profile,
)
from multi_agent_app.nodes.contact_node import contact_node, CONTACT_TOOLS as contact_tools
from multi_agent_app.nodes.profile_node import profile_node, TOOLS as profile_tools
from multi_agent_app.state import AgentState
from multi_agent_app.nodes.supervisor_node import supervisor_node
from multi_agent_app.nodes.chatter_node import chatter_node
# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)

# Nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("profile_node", profile_node)
builder.add_node("profile_tools", ToolNode(profile_tools))
builder.add_node("contact_node", contact_node)
builder.add_node("contact_tools", ToolNode(contact_tools))
builder.add_node("chatter_node", chatter_node)

# Edges
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor",
                              route_next,
                           {
                               "profile_node": "profile_node",
                               "contact_node": "contact_node",
                               "chatter_node": "chatter_node",
                               "end": END,
                           })

# Conditional edge: stay in the loop while the model calls tools;
# exit when the model produces a final answer.
builder.add_conditional_edges(
    "profile_node",
    should_continue_profile,
    {
        "profile_tools": "profile_tools",
        "end": END,
    },
)
builder.add_conditional_edges(
    "contact_node",
    should_continue_contact,
    {
        "contact_tools": "contact_tools",
        "end": END,
    },
)

# After every tool execution, hand control back to the agent.
builder.add_edge("profile_tools", "profile_node")
builder.add_edge("contact_tools", "contact_node")
builder.add_edge("chatter_node", "supervisor")


# ---------------------------------------------------------------------------
# Compile – this is what LangGraph Studio imports.
# --------------------------------------------------------------------------
graph = builder.compile(debug=True)
graph.name = "Multi-Agent App"
