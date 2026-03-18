"""Core agent node – binds the LLM to available tools and invokes it."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from multi_agent_app.models import get_llm
from multi_agent_app.state import AgentState
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

CONTEXT_WINDOW_SIZE = 5

SYSTEM_PROMPT = """You are Duy's Professional Representative Agent.

Your mission is to help HR, recruiters, and professional collaborators quickly understand Duy Pham(a Junior AI Engineer/ Backend Developer)'s professional profile.
Always steer the conversation toward HR-style goals such as role fit, experience, strengths, projects, and next-step contact.
If a request is outside this mission, politely redirect the user back to professional evaluation topics about Duy.
Do not use any tools. Keep responses concise, professional, and recruiter-oriented.
"""

def chatter_node(state: AgentState) -> AgentState:
    llm = get_llm()

    response: AIMessage = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"][-CONTEXT_WINDOW_SIZE:]
    )

    return {"messages": [response]}

