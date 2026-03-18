"""
worker/main.py
--------------
Worker process entry-point.

Architecture
------------
- On startup: recover orphaned jobs, warm the graph singleton.
- Spawns `WORKER_CONCURRENCY` async tasks, each running an infinite consume loop.
- Graceful shutdown via SIGINT / SIGTERM: drains in-flight tasks then exits.

Run
---
    python -m worker.main
    # or inside Docker:
    python -m worker.main
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

# Ensure production/ root is on sys.path when run as `python -m worker.main`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import get_settings
from core.redis_client import close_redis, get_redis
from graph import get_graph
from services.queue import RedisQueue
from services.state_store import StateStore
from worker.runner import execute_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("worker")


async def _consume_loop(
    queue: RedisQueue,
    redis,
    state_store: StateStore,
    shutdown_event: asyncio.Event,
    worker_id: int,
) -> None:
    """Single consumer loop — runs until shutdown_event is set."""
    log = logging.getLogger(f"worker.consumer.{worker_id}")
    log.info("Consumer %d started", worker_id)

    while not shutdown_event.is_set():
        try:
            payload = await queue.pop_job(timeout=5)
        except Exception as exc:
            log.error("Error popping from queue: %s", exc)
            await asyncio.sleep(1)
            continue

        if payload is None:
            # Timeout — loop back to check shutdown_event
            continue

        raw_payload = json.dumps(
            {
                "job_id": payload["job_id"],
                "session_id": payload["session_id"],
                "user_message": payload["user_message"],
                "metadata": payload.get("metadata") or {},
            }
        )

        log.info("Processing job %s", payload.get("job_id"))
        try:
            await execute_job(payload, redis, state_store)
        except Exception as exc:
            log.exception("Unhandled error in execute_job: %s", exc)
        finally:
            # Always ack so the item leaves the processing list
            await queue.ack_job(raw_payload)

    log.info("Consumer %d shutting down", worker_id)


async def main() -> None:
    settings = get_settings()
    redis = await get_redis()

    queue = RedisQueue(redis)
    state_store = StateStore(redis)

    # Recover orphaned jobs from a previous crash
    await queue.requeue_orphaned()

    # Warm the graph singleton before accepting jobs
    logger.info("Warming LangGraph singleton…")
    get_graph()
    logger.info("Graph ready.")

    shutdown_event = asyncio.Event()

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("Received %s — initiating graceful shutdown…", sig.name)
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except (NotImplementedError, AttributeError):
            # Windows does not support add_signal_handler for all signals
            signal.signal(sig, lambda s, f: shutdown_event.set())

    tasks = [
        asyncio.create_task(
            _consume_loop(queue, redis, state_store, shutdown_event, i)
        )
        for i in range(settings.worker_concurrency)
    ]

    logger.info(
        "Worker running with %d consumer(s). Press Ctrl+C to stop.",
        settings.worker_concurrency,
    )

    await shutdown_event.wait()
    logger.info("Waiting for in-flight tasks…")
    await asyncio.gather(*tasks, return_exceptions=True)
    await close_redis()
    logger.info("Worker stopped.")


if __name__ == "__main__":
    asyncio.run(main())
