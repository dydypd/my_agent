"""
api/routes/ws.py
----------------
WebSocket endpoint for bi-directional real-time communication.

WS /ws/{session_id}
    ← Client sends: {"user_message": "Hello"}
    → Server streams: {"event": "status", "status": "running"}
                      {"event": "retry",  "attempt": 1, "error": "..."}   (if retry)
                      {"event": "done",   "final_answer": "...", "status": "done"}
                   OR {"event": "error",  "error": "...", "status": "failed"}

The connection stays open after each exchange so the client can send
multiple messages in sequence without reconnecting.

Client usage (JavaScript):
--------------------------
    const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.event === "done") console.log(msg.final_answer);
    };
    ws.send(JSON.stringify({ user_message: "Hello!" }));
"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.dependencies import get_redis_dep, get_state_store
from models.job import JobRequest, JobResult
from services.queue import RedisQueue
from services.state_store import StateStore

logger = logging.getLogger(__name__)
router_ws = APIRouter(tags=["websocket"])

_WS_RECV_TIMEOUT = 300.0  # seconds to wait for next client message


@router_ws.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """
    Bi-directional WebSocket for a single session.

    Each text message the client sends triggers a new job.  The server
    streams progress events back over the same connection and remains open
    for subsequent messages.
    """
    await websocket.accept()
    logger.info("WS connection opened for session %s", session_id)

    redis: aioredis.Redis = await get_redis_dep()
    store = StateStore(redis)
    queue = RedisQueue(redis)

    try:
        while True:
            # ── Wait for next client message ───────────────────────────────
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=_WS_RECV_TIMEOUT
                )
            except asyncio.TimeoutError:
                await websocket.send_text(
                    json.dumps({"event": "ping", "message": "still connected"})
                )
                continue

            try:
                body = json.loads(raw)
                user_message: str = body["user_message"]
            except (json.JSONDecodeError, KeyError):
                await websocket.send_text(
                    json.dumps(
                        {"event": "error", "error": "Invalid payload. Expected {\"user_message\": \"...\"}"}
                    )
                )
                continue

            # ── Enqueue job ────────────────────────────────────────────────
            req = JobRequest(session_id=session_id, user_message=user_message)
            job_result = JobResult(
                session_id=session_id,
                user_message=user_message,
            )
            job_id = job_result.job_id
            await store.save_job_result(job_result)
            await queue.push_job(req, job_id)
            logger.info("WS enqueued job %s", job_id)

            # Acknowledge immediately
            await websocket.send_text(
                json.dumps({"event": "queued", "job_id": job_id})
            )

            # ── Subscribe to job events and forward to WS ──────────────────
            channel = f"job:{job_id}:events"
            pubsub = redis.pubsub()
            await pubsub.subscribe(channel)

            try:
                async for raw_msg in pubsub.listen():
                    if raw_msg["type"] != "message":
                        continue
                    try:
                        payload = json.loads(raw_msg["data"])
                    except json.JSONDecodeError:
                        continue

                    await websocket.send_text(json.dumps(payload))

                    # Stop listening when job reaches a terminal state
                    if payload.get("event") in ("done", "error"):
                        break
            finally:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()

    except WebSocketDisconnect:
        logger.info("WS disconnected for session %s", session_id)
    except Exception as exc:
        logger.exception("WS error for session %s: %s", session_id, exc)
        try:
            await websocket.send_text(
                json.dumps({"event": "error", "error": str(exc)})
            )
        except Exception:
            pass
    finally:
        logger.info("WS connection closed for session %s", session_id)
