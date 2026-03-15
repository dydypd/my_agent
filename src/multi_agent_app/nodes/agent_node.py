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
