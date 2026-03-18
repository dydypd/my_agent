"""Core agent node – binds the LLM to available tools and invokes it."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from multi_agent_app.models import get_llm
from multi_agent_app.state import AgentState
from langchain_core.messages import SystemMessage
import inspect
from typing import List, Callable
from multi_agent_app.tools.profile_tools import profile_retriever
# All tools available to the agent
TOOLS = [profile_retriever]

# Only keep a short rolling window of chat history for each LLM call.
CONTEXT_WINDOW_SIZE = 5


def get_tools_description(tools: List[Callable]) -> str:
    """Return a short description of the tools."""
    descriptions = []
    for tool in tools:
        name = tool.name
        doc = getattr(tool, "description", inspect.getdoc(tool)) or "No description available."
        descriptions.append(f"- {name}: {doc}")
    return "\n".join(descriptions)


SYSTEM_PROMPT = """You are Duy's Professional Representative Agent.

Your role is to represent Duy Pham primarily in HR and recruiter conversations about his professional background.
Help users evaluate Duy's role fit, experience, strengths, and relevant projects.
You do NOT directly know Duy's profile.
Instead, you must retrieve information using the available tools before answering.
You have access to the following tools:
{tools_description}

## Decision rules
- ALWAYS call `profile_retriever` before answering any question about a person's profile.
- If the user asks something unrelated to professional evaluation of Duy, politely redirect and stay on HR/recruiter topic.
- Never fabricate profile data — only use what the tool returns.
""".format(tools_description=get_tools_description(TOOLS))


def profile_node(state: AgentState) -> AgentState:
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    recent_messages = state["messages"][-CONTEXT_WINDOW_SIZE:]

    response: AIMessage = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + recent_messages
    )

    return {"messages": [response]}
