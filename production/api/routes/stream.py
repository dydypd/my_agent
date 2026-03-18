"""
api/routes/stream.py
--------------------
SSE (Server-Sent Events) endpoint.

GET /jobs/{job_id}/stream
    → Subscribes to Redis Pub/Sub channel `job:<job_id>:events`
    → Streams events to the browser as text/event-stream
    → Closes the connection automatically on "done" or "error" event

If the job is already completed when the client connects, the handler
reads the stored result from Redis and sends a single event immediately
(no stale Pub/Sub subscription).

Client usage (browser):
-----------------------
    const es = new EventSource(`/jobs/${jobId}/stream`);
    es.addEventListener("done", e => {
        const { final_answer } = JSON.parse(e.data);
        es.close();
    });
    es.addEventListener("error_event", e => {
        console.error(JSON.parse(e.data).error);
        es.close();
    });

Note: the SSE event named "error" is reserved by the browser EventSource
spec, so we use "error_event" for application-level errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import RedisDep, StateStoreDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])

_HEARTBEAT_INTERVAL = 15  # seconds — keeps connections alive through proxies


def _fmt(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _sse_generator(
    job_id: str,
    redis: aioredis.Redis,
    store,
) -> AsyncGenerator[str, None]:
    """
    Core SSE generator.

    1. If job is already done/failed → emit one event and return.
    2. Otherwise subscribe to Pub/Sub and yield events until terminal state.
    """
    # ── Fast-path: job already finished ───────────────────────────────────
    result = await store.load_job_result(job_id)
    if result is None:
        yield _fmt("error_event", {"error": f"Job '{job_id}' not found."})
        return

    if result.status == "done":
        yield _fmt("done", {"final_answer": result.final_answer, "status": "done"})
        return

    if result.status == "failed":
        yield _fmt("error_event", {"error": result.error, "status": "failed"})
        return

    # ── Subscribe and stream ───────────────────────────────────────────────
    channel = f"job:{job_id}:events"
    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)
    logger.debug("SSE subscribed to %s", channel)

    try:
        # Poll pubsub with timeout so we can emit heartbeat comments even when
        # no job events are published for a while (prevents Heroku H15 idle timeout).
        while True:
            raw = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=_HEARTBEAT_INTERVAL,
            )

            if raw is None:
                yield ": heartbeat\n\n"
                continue

            try:
                payload: dict = json.loads(raw["data"])
            except json.JSONDecodeError:
                continue

            event_name: str = payload.pop("event", "message")
            # Remap "error" → "error_event" to avoid browser spec collision
            if event_name == "error":
                event_name = "error_event"

            yield _fmt(event_name, payload)

            # Close stream once terminal event received
            if event_name in ("done", "error_event"):
                break
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        logger.debug("SSE unsubscribed from %s", channel)


router_stream = APIRouter(tags=["stream"])


@router_stream.get(
    "/jobs/{job_id}/stream",
    summary="Stream job events via SSE",
    response_class=StreamingResponse,
)
async def stream_job(
    job_id: str,
    redis: RedisDep,
    store: StateStoreDep,
) -> StreamingResponse:
    """
    Subscribe to job progress events via Server-Sent Events.
    The stream closes automatically when the job reaches a terminal state.
    """
    return StreamingResponse(
        _sse_generator(job_id, redis, store),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",          # disable Nginx buffering
            "Connection": "keep-alive",
        },
    )
