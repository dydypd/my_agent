"""
services/queue.py
-----------------
Redis List-based reliable job queue.

Design
------
LPUSH  ─► agent:jobs             (producer pushes to head)
BRPOPLPUSH agent:jobs agent:jobs:processing  (consumer atomically moves tail item
                                              to processing list)
LREM   agent:jobs:processing     (consumer acks after success)

The processing list acts as a crash-safety net: if a worker dies mid-job,
a supervisor can LRANGE agent:jobs:processing and re-enqueue orphaned items.
"""

from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis

from core.config import get_settings
from models.job import JobRequest, JobResult

logger = logging.getLogger(__name__)


class RedisQueue:
    """Producer + consumer for the agent job queue."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._settings = get_settings()
        self._q = self._settings.job_queue_name
        self._pq = self._settings.job_processing_queue_name

    # ── Producer ───────────────────────────────────────────────────────────

    async def push_job(self, request: JobRequest, job_id: str) -> None:
        """Serialize job payload and LPUSH onto the queue."""
        payload = json.dumps(
            {
                "job_id": job_id,
                "session_id": request.session_id,
                "user_message": request.user_message,
                "metadata": request.metadata,
            }
        )
        await self._redis.lpush(self._q, payload)
        logger.info("Pushed job %s to queue %s", job_id, self._q)

    # ── Consumer ───────────────────────────────────────────────────────────

    async def pop_job(self, timeout: int = 5) -> dict | None:
        """
        Blocking pop from the tail of agent:jobs into agent:jobs:processing.

        Returns the parsed dict, or None if timeout elapsed without a message.
        The item remains in the processing list until `ack_job` is called.
        """
        raw = await self._redis.brpoplpush(self._q, self._pq, timeout=timeout)
        if raw is None:
            return None
        data = json.loads(raw)
        logger.debug("Popped job %s from queue", data.get("job_id"))
        return data

    async def ack_job(self, raw_payload: str) -> None:
        """Remove a job from the processing list after successful completion."""
        removed = await self._redis.lrem(self._pq, 1, raw_payload)
        if removed == 0:
            logger.warning("ack_job: payload not found in processing list — already removed?")

    async def requeue_orphaned(self) -> int:
        """
        Move all items still in the processing list back to the main queue.
        Call this on worker startup to recover from a previous crash.
        Returns the number of items re-queued.
        """
        items = await self._redis.lrange(self._pq, 0, -1)
        count = 0
        for item in items:
            await self._redis.lpush(self._q, item)
            await self._redis.lrem(self._pq, 1, item)
            count += 1
        if count:
            logger.info("Re-queued %d orphaned job(s) from processing list", count)
        return count
