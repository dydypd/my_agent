"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Supported LLM provider identifiers
PROVIDER_NVIDIA = "nvidia"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
SUPPORTED_PROVIDERS = (PROVIDER_NVIDIA, PROVIDER_OPENAI, PROVIDER_GEMINI)


@dataclass
class Configuration:
    """Central configuration object for the agent application."""

    # Provider selection: "nvidia" (default), "openai", or "gemini"
    provider: str = PROVIDER_NVIDIA

    # LLM settings – default model targets Nvidia NIM
    model_name: str = "meta/llama-3.1-70b-instruct"
    temperature: float = 0.0
    max_tokens: int = 4096

    # Nvidia NIM base URL (override to point at a self-hosted NIM instance)
    nvidia_base_url: str = field(default="https://integrate.api.nvidia.com/v1")

    # LangSmith / tracing
    langchain_project: str = "multi_agent_app"

    @classmethod
    def from_env(cls) -> "Configuration":
        """Construct a Configuration instance from the current environment."""
        provider = os.getenv("PROVIDER", PROVIDER_NVIDIA).lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported PROVIDER '{provider}'. "
                f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}"
            )

        # Per-provider default model names
        default_models = {
            PROVIDER_NVIDIA: "meta/llama-3.1-70b-instruct",
            PROVIDER_OPENAI: "gpt-4o-mini",
            PROVIDER_GEMINI: "gemini-1.5-flash",
        }

        return cls(
            provider=provider,
            model_name=os.getenv("MODEL_NAME", default_models[provider]),
            temperature=float(os.getenv("TEMPERATURE", "0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            nvidia_base_url=os.getenv(
                "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
            ),
            langchain_project=os.getenv("LANGCHAIN_PROJECT", "multi_agent_app"),
        )
