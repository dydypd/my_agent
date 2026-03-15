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
