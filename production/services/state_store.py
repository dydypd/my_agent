"""
services/state_store.py
------------------------
Persistent session state and job result storage in Redis.

Keys
----
session:<session_id>:state  →  JSON-serialised AgentState (with TTL)
job:<job_id>:result         →  JSON-serialised JobResult   (with TTL)

AgentState serialisation
------------------------
LangChain messages are not directly JSON-serialisable, so we use
`messages_to_dict` / `messages_from_dict` from langchain_core.messages.
All other AgentState fields are plain JSON-friendly Python objects.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from core.config import get_settings
from models.job import JobResult

logger = logging.getLogger(__name__)


class StateStore:
    """Handles serialisation, storage, and retrieval of state/results."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._settings = get_settings()
        self._ttl = self._settings.job_ttl_seconds

    # ── Session State ──────────────────────────────────────────────────────

    async def save_session_state(
        self, session_id: str, state: dict[str, Any]
    ) -> None:
        """Persist AgentState dict to Redis with TTL."""
        serialisable = _serialise_state(state)
        key = _session_key(session_id)
        await self._redis.set(key, json.dumps(serialisable), ex=self._ttl)
        logger.debug("Saved session state for %s", session_id)

    async def load_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Load and deserialise a previously saved AgentState, or None."""
        key = _session_key(session_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        return _deserialise_state(data)

    # ── Job Result ─────────────────────────────────────────────────────────

    async def save_job_result(self, result: JobResult) -> None:
        """Persist a JobResult (any status) to Redis with TTL."""
        key = _job_key(result.job_id)
        await self._redis.set(
            key,
            result.model_dump_json(),
            ex=self._ttl,
        )
        logger.debug("Saved job result %s (status=%s)", result.job_id, result.status)

    async def load_job_result(self, job_id: str) -> JobResult | None:
        """Load a JobResult from Redis, or None if it doesn't exist."""
        key = _job_key(job_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return JobResult.model_validate_json(raw)


# ── Helpers ────────────────────────────────────────────────────────────────


def _session_key(session_id: str) -> str:
    return f"session:{session_id}:state"


def _job_key(job_id: str) -> str:
    return f"job:{job_id}:result"


def _serialise_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert AgentState to a JSON-safe dict."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k == "messages" and isinstance(v, list):
            # Serialise LangChain message objects
            out[k] = messages_to_dict(
                [m for m in v if isinstance(m, BaseMessage)]
            )
        else:
            # Everything else is assumed JSON-safe (str, int, list[str], dict, None)
            out[k] = v
    return out


def _deserialise_state(data: dict[str, Any]) -> dict[str, Any]:
    """Restore AgentState from a JSON-safe dict."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k == "messages" and isinstance(v, list):
            out[k] = messages_from_dict(v)
        else:
            out[k] = v
    return out
