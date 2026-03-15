"""LLM factory helpers."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .configuration import Configuration


def get_llm(config: Configuration | None = None) -> ChatOpenAI:
    """
    Return an initialised ChatOpenAI instance.

    Parameters
    ----------
    config:
        Optional :class:`Configuration` object.  When *None* a default
        instance is created from the current environment.
    """
    cfg = config or Configuration.from_env()
    return ChatOpenAI(
        model=cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
