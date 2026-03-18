"""
core/config.py
--------------
Central settings loaded from environment variables via pydantic-settings.
All services import `get_settings()` to share one instance.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Redis ──────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Queue ──────────────────────────────────────────────────
    job_queue_name: str = "agent:jobs"
    job_processing_queue_name: str = "agent:jobs:processing"
    job_ttl_seconds: int = 3600  # result TTL in Redis

    # ── Worker ─────────────────────────────────────────────────
    worker_concurrency: int = 4  # async task slots per worker process
    max_retries: int = 3
    retry_backoff_base: float = 2.0  # seconds; sleep = base ** attempt
    graph_invoke_timeout: float = 120.0  # seconds per graph.ainvoke call

    # ── API ────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── LLM Provider (passed through to the graph) ─────────────
    provider: str = "nvidia"
    model_name: str = "deepseek-ai/deepseek-v3.1-terminus"
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    openai_api_key: str = ""
    google_api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096

    # ── LangSmith ──────────────────────────────────────────────
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "my_enterprise_agent"

    # ── Qdrant ─────────────────────────────────────────────────
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "recruiter_profile"
    embedding_provider: str = "nvidia"
    rag_top_k: int = 5
    rag_score_threshold: float = 0.0

    # ── Discord ────────────────────────────────────────────────
    discord_webhook_url: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton Settings instance (cached after first call)."""
    return Settings()
