"""LLM factory helpers."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .configuration import (
    PROVIDER_GEMINI,
    PROVIDER_NVIDIA,
    PROVIDER_OPENAI,
    Configuration,
)


def get_llm(config: Configuration | None = None) -> BaseChatModel:
    """
    Return an initialised chat-model instance for the configured provider.

    The default provider is **Nvidia NIM**.  Set the ``PROVIDER`` environment
    variable to ``"openai"`` or ``"gemini"`` to use those providers instead.

    Parameters
    ----------
    config:
        Optional :class:`Configuration` object.  When *None* a default
        instance is created from the current environment.
    """
    cfg = config or Configuration.from_env()

    if cfg.provider == PROVIDER_NVIDIA:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA  # noqa: PLC0415

        return ChatNVIDIA(
            model=cfg.model_name,
            base_url=cfg.nvidia_base_url,
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_tokens,
        )

    if cfg.provider == PROVIDER_OPENAI:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        return ChatOpenAI(
            model=cfg.model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if cfg.provider == PROVIDER_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

        return ChatGoogleGenerativeAI(
            model=cfg.model_name,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens,
        )

    # Fallback guard: reachable if a Configuration is created directly
    # with an unsupported provider string (bypassing from_env() validation).
    raise ValueError(f"Unsupported provider: '{cfg.provider}'")
