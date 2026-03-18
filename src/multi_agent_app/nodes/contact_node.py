"""Core agent node – binds the LLM to available tools and invokes it."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, ToolMessage

from multi_agent_app.models import get_llm
from multi_agent_app.state import AgentState
from langchain_core.messages import SystemMessage
import inspect
from typing import List, Callable
from multi_agent_app.tools.contact_tools import notify_me_discord, get_time
from multi_agent_app.tools.profile_tools import profile_retriever
from langchain_core.messages import HumanMessage
# All tools available to the agent
CONTACT_TOOLS = [notify_me_discord, profile_retriever, get_time]
CONTEXT_WINDOW_SIZE = 5
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

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
Guide users toward concrete next-step contact in a hiring or professional collaboration context.
You do NOT directly know Duy's contact information.
Instead, you must retrieve information using the available tools before answering.
You have access to the following tools:
{tools_description}

## Decision rules
- Require the HR/recruiter to provide JD as a link before any contact handoff.
- If no JD link is provided yet, ask for the JD link and do not proceed to contact handoff.
- Only call `notify_me_discord` after a JD link is available.
- If the user asks something unrelated to hiring/professional contact for Duy, politely redirect and stay on topic.
- Never fabricate contact data — only use what the tool returns.
- If you have enough information to answer the user's question, call 'profile_retriever' to get Duy's profile and then respond to the user.
""".format(tools_description=get_tools_description(CONTACT_TOOLS))


def _extract_text_content(content: object) -> str:
    """Flatten message content into text for simple URL detection."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _has_jd_link(messages: list) -> bool:
    """Return True when any human message in the conversation contains a URL."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            text = _extract_text_content(message.content)
            if URL_PATTERN.search(text):
                return True
    return False


def _has_notify_result(messages: list) -> bool:
    """Return True when notify_me_discord has already produced a tool result."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and getattr(message, "name", "") == "notify_me_discord":
            return True
    return False


def contact_node(state: AgentState) -> AgentState:
    llm = get_llm()
    llm_with_tools = llm.bind_tools(CONTACT_TOOLS)
    all_messages = state["messages"]
    recent_messages = all_messages[-CONTEXT_WINDOW_SIZE:]

    if not _has_jd_link(all_messages) and not _has_notify_result(all_messages):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Mình chỉ hỗ trợ chuyển liên hệ sau khi bạn gửi JD dưới dạng link. "
                        "Vui lòng gửi link JD/job post để mình tiếp tục."
                    )
                )
            ]
        }

    response: AIMessage = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + recent_messages
    )

    return {"messages": [response]}

