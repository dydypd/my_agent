"""
worker/runner.py
----------------
Core logic for a single job execution.

Responsibilities
----------------
1. Load prior session state from Redis (or initialise blank state)
2. Append the new human message to the messages list
3. Call graph.ainvoke(state) with retry + timeout
4. Persist the updated session state
5. Publish SSE events to Redis Pub/Sub channel
6. Return the final_answer string
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import redis.asyncio as aioredis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from core.config import get_settings
from graph import get_graph
from models.job import JobResult
from services.state_store import StateStore
from worker.retry import run_with_retry

logger = logging.getLogger(__name__)


def _sse_channel(job_id: str) -> str:
    return f"job:{job_id}:events"


async def _publish(redis: aioredis.Redis, channel: str, event: str, data: dict) -> None:
    payload = json.dumps({"event": event, **data})
    await redis.publish(channel, payload)


def _extract_final_answer(result_state: dict[str, Any]) -> str:
    """
    Pull the final answer out of a completed AgentState.

    Priority:
        1. result_state["final_answer"] if set
        2. Last AIMessage in result_state["messages"]
        3. Fallback string
    """
    if result_state.get("final_answer"):
        return str(result_state["final_answer"])

    messages: list[BaseMessage] = result_state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)

    return "(no response)"


async def _invoke_graph(state: dict[str, Any]) -> dict[str, Any]:
    """Thin wrapper so run_with_retry can call it."""
    graph = get_graph()
    return await graph.ainvoke(state)  # type: ignore[return-value]


async def execute_job(
    job_payload: dict,
    redis: aioredis.Redis,
    state_store: StateStore,
) -> JobResult:
    """
    Execute one job end-to-end.

    Parameters
    ----------
    job_payload : dict
        Raw dict from the queue: {job_id, session_id, user_message, metadata}
    redis : aioredis.Redis
        Used for Pub/Sub event publishing.
    state_store : StateStore
        Reads and writes session state + job result.

    Returns
    -------
    JobResult with status "done" or "failed".
    """
    settings = get_settings()
    job_id: str = job_payload["job_id"]
    session_id: str = job_payload["session_id"]
    user_message: str = job_payload["user_message"]
    metadata: dict = job_payload.get("metadata") or {}
    channel = _sse_channel(job_id)

    # ── Load current job result so we can update it ────────────────────────
    job_result = await state_store.load_job_result(job_id)
    if job_result is None:
        logger.error("Job %s not found in state store — skipping", job_id)
        return JobResult(
            job_id=job_id,
            session_id=session_id,
            status="failed",
            error="Job not found in state store",
        )

    # Mark running
    job_result = job_result.mark_running()
    await state_store.save_job_result(job_result)
    await _publish(redis, channel, "status", {"status": "running"})

    # ── Load prior session state ───────────────────────────────────────────
    prior_state = await state_store.load_session_state(session_id) or {}
    prior_messages: list[BaseMessage] = list(prior_state.get("messages") or [])

    # ── Build input state ─────────────────────────────────────────────────
    input_state: dict[str, Any] = {
        **prior_state,
        "messages": [*prior_messages, HumanMessage(content=user_message)],
        "metadata": {**metadata},
        # Reset per-run transient fields so they don't bleed across turns
        "next": None,
        "current_agent": None,
        "error": None,
        "final_answer": None,
        "step_count": prior_state.get("step_count", 0),
        "scratchpad": "",
    }

    # ── Execute graph with retry ───────────────────────────────────────────
    try:
        result_state, attempts = await run_with_retry(
            _invoke_graph,
            input_state,
            attempt_callback=lambda i, exc: asyncio.ensure_future(
                _publish(
                    redis,
                    channel,
                    "retry",
                    {"attempt": i + 1, "error": str(exc)},
                )
            ),
        )

        final_answer = _extract_final_answer(result_state)

        # ── Persist updated session state ──────────────────────────────────
        await state_store.save_session_state(session_id, result_state)

        # ── Update job result ──────────────────────────────────────────────
        job_result = job_result.mark_done(final_answer)
        job_result = job_result.model_copy(update={"retry_count": attempts})
        await state_store.save_job_result(job_result)

        # ── Publish done event for SSE / WS listeners ──────────────────────
        await _publish(
            redis,
            channel,
            "done",
            {"final_answer": final_answer, "status": "done"},
        )
        logger.info("Job %s completed (attempts=%d)", job_id, attempts + 1)

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Job %s failed permanently: %s", job_id, error_msg)
        job_result = job_result.mark_failed(error_msg, settings.max_retries)
        await state_store.save_job_result(job_result)
        await _publish(
            redis,
            channel,
            "error",
            {"error": error_msg, "status": "failed"},
        )

    return job_result
