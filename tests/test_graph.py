"""Smoke tests for the compiled graph."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage


def test_graph_importable() -> None:
    """The graph module must be importable without side-effects."""
    from multi_agent_app.graph import graph  # noqa: PLC0415

    assert graph is not None


def test_graph_has_nodes() -> None:
    """The compiled graph must contain the expected node names."""
    from multi_agent_app.graph import graph  # noqa: PLC0415

    node_names = set(graph.nodes.keys())
    assert "agent" in node_names
    assert "tools" in node_names


def test_configuration_defaults_to_nvidia() -> None:
    """Default provider must be Nvidia NIM."""
    from multi_agent_app.configuration import Configuration  # noqa: PLC0415

    cfg = Configuration()
    assert cfg.provider == "nvidia"
    assert cfg.model_name == "meta/llama-3.1-70b-instruct"
    assert cfg.nvidia_base_url == "https://integrate.api.nvidia.com/v1"


def test_configuration_from_env_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env() with PROVIDER=nvidia returns correct NIM defaults."""
    monkeypatch.setenv("PROVIDER", "nvidia")
    monkeypatch.delenv("MODEL_NAME", raising=False)

    from multi_agent_app.configuration import Configuration  # noqa: PLC0415

    cfg = Configuration.from_env()
    assert cfg.provider == "nvidia"
    assert cfg.model_name == "meta/llama-3.1-70b-instruct"


def test_configuration_from_env_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env() with PROVIDER=openai returns correct OpenAI defaults."""
    monkeypatch.setenv("PROVIDER", "openai")
    monkeypatch.delenv("MODEL_NAME", raising=False)

    from multi_agent_app.configuration import Configuration  # noqa: PLC0415

    cfg = Configuration.from_env()
    assert cfg.provider == "openai"
    assert cfg.model_name == "gpt-4o-mini"


def test_configuration_from_env_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env() with PROVIDER=gemini returns correct Gemini defaults."""
    monkeypatch.setenv("PROVIDER", "gemini")
    monkeypatch.delenv("MODEL_NAME", raising=False)

    from multi_agent_app.configuration import Configuration  # noqa: PLC0415

    cfg = Configuration.from_env()
    assert cfg.provider == "gemini"
    assert cfg.model_name == "gemini-1.5-flash"


def test_configuration_from_env_custom_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """MODEL_NAME env var overrides per-provider default."""
    monkeypatch.setenv("PROVIDER", "nvidia")
    monkeypatch.setenv("MODEL_NAME", "meta/llama-3.1-8b-instruct")

    from multi_agent_app.configuration import Configuration  # noqa: PLC0415

    cfg = Configuration.from_env()
    assert cfg.model_name == "meta/llama-3.1-8b-instruct"


def test_configuration_from_env_invalid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env() raises ValueError for unsupported providers."""
    monkeypatch.setenv("PROVIDER", "unknown_provider")

    from multi_agent_app.configuration import Configuration  # noqa: PLC0415

    with pytest.raises(ValueError, match="Unsupported PROVIDER"):
        Configuration.from_env()


def test_get_llm_returns_nvidia_type() -> None:
    """get_llm() with nvidia provider returns a ChatNVIDIA instance."""
    from langchain_nvidia_ai_endpoints import ChatNVIDIA  # noqa: PLC0415

    from multi_agent_app.configuration import Configuration  # noqa: PLC0415
    from multi_agent_app.models import get_llm  # noqa: PLC0415

    cfg = Configuration(provider="nvidia", model_name="meta/llama-3.1-70b-instruct")
    llm = get_llm(cfg)
    assert isinstance(llm, ChatNVIDIA)


def test_get_llm_returns_openai_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_llm() with openai provider returns a ChatOpenAI instance."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    from multi_agent_app.configuration import Configuration  # noqa: PLC0415
    from multi_agent_app.models import get_llm  # noqa: PLC0415

    cfg = Configuration(provider="openai", model_name="gpt-4o-mini", temperature=0.5, max_tokens=512)
    llm = get_llm(cfg)
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"
    assert llm.temperature == 0.5
    assert llm.max_tokens == 512


def test_get_llm_returns_gemini_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_llm() with gemini provider returns a ChatGoogleGenerativeAI instance."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

    from multi_agent_app.configuration import Configuration  # noqa: PLC0415
    from multi_agent_app.models import get_llm  # noqa: PLC0415

    cfg = Configuration(provider="gemini", model_name="gemini-1.5-flash", temperature=0.7, max_tokens=1024)
    llm = get_llm(cfg)
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == "gemini-1.5-flash"
    assert llm.temperature == 0.7
    assert llm.max_output_tokens == 1024


@pytest.mark.skip(reason="Requires a live Nvidia NIM API key – run manually")
def test_graph_invoke_returns_messages() -> None:
    """End-to-end: the graph should return at least one AI message."""
    from langchain_core.messages import AIMessage  # noqa: PLC0415
    from multi_agent_app.graph import graph  # noqa: PLC0415

    result = graph.invoke({"messages": [HumanMessage(content="Hello!")]})
    assert len(result["messages"]) >= 2
    assert isinstance(result["messages"][-1], AIMessage)
