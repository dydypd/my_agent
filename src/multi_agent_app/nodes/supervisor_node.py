"""Supervisor node – routes user request to the correct sub-agent."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from multi_agent_app.models import get_llm
from multi_agent_app.state import AgentState
from pydantic import BaseModel


class RouteDecision(BaseModel):
   next: Literal["profile_node", "contact_node", "chatter_node", "end"] = Field(
      description="The next agent to route the user request to"
   )

SUPERVISOR_PROMPT = """You are a supervisor agent managing a team of specialized sub-agents. 
Your job is to analyze the user's input and route it to the most appropriate sub-agent based on their capabilities.

Available agents and their specific use cases:
1. `profile_node`
   - Use this when the user asks questions about Duy's professional background.
   - Topics include: skills, work experience, education, projects, CV/resume details, and professional capabilities.
   - Example triggers: "What are his skills?", "Where did he work?", "Tell me about his education".

2. `contact_node`
   - Use this when the user expresses a desire to contact, reach out, or hire Duy.
   - Topics include: sending an email, scheduling a meeting, getting contact info, or notifying him via Discord.
   - Example triggers: "How can I contact him?", "I want to schedule an interview", "Send him a message".

3. `chatter_node`
   - Use this as a fallback for general conversation and chitchat.
   - Topics include: greetings, casual talk, or questions completely unrelated to Duy's professional profile or contacting him.
   - Example triggers: "Hello", "How are you?", "What is 2+2?".

4. `end`
   - Use this ONLY when the conversation has reached a natural conclusion or the user explicitly wants to end it.
   - Example triggers: "Goodbye", "Thanks for the help, bye", "That's all I needed".

Decision Guidelines:
- Carefully analyze the INTENT of the user's message.
- If the user explicitly wants to contact or set up a meeting, route to `contact_node`.
- If the user is just asking about Duy's skillset or experience, route to `profile_node`.
- Always default to `chatter_node` if the request is just casual chat and doesn't fit `profile_node` or `contact_node`.

Always respond with the JSON format.
{
    "next": "profile_node" | "contact_node" | "chatter_node" | "end",
    "reason": "The reason for the decision"
}
"""


def supervisor_node(state: AgentState) -> AgentState:
   messages = state.get("messages") or []
   metadata = dict(state.get("metadata") or {})

   latest_human_index = None
   latest_human: HumanMessage | None = None
   for index in range(len(messages) - 1, -1, -1):
      if isinstance(messages[index], HumanMessage):
         latest_human_index = index
         latest_human = messages[index]
         break

   if latest_human is None or latest_human_index is None:
      return {
         "next": "end",
         "current_agent": "supervisor",
         "step_count": state.get("step_count", 0) + 1,
         "metadata": metadata,
      }

   # Prevent routing the same user turn repeatedly when control returns
   # from a downstream node without a new HumanMessage.
   last_routed_human_index = metadata.get("last_routed_human_index")
   if last_routed_human_index == latest_human_index:
      return {
         "next": "end",
         "current_agent": "supervisor",
         "step_count": state.get("step_count", 0) + 1,
         "metadata": metadata,
      }

   llm = get_llm()
   structured_llm = llm.with_structured_output(RouteDecision)
   decision = structured_llm.invoke(
      [
         SystemMessage(content=SUPERVISOR_PROMPT),
         HumanMessage(content=latest_human.content),
      ]
   )

   route_history = list(metadata.get("route_history") or [])
   route_history.append(
      {
         "human_message_index": latest_human_index,
         "selected_node": decision.next,
      }
   )
   metadata["last_routed_human_index"] = latest_human_index
   metadata["route_history"] = route_history

   print("Supervisor decision:", decision)
   return {
      "next": decision.next,
      "current_agent": "supervisor",
      "step_count": state.get("step_count", 0) + 1,
      "metadata": metadata,
   }