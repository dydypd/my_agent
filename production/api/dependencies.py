"""
api/dependencies.py
-------------------
FastAPI dependency injection: provides shared async clients
(Redis, Queue, StateStore) to route handlers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
import redis.asyncio as aioredis

from core.redis_client import get_redis as _get_redis
from services.queue import RedisQueue
from services.state_store import StateStore


async def get_redis_dep() -> aioredis.Redis:
    return await _get_redis()


async def get_queue(
    redis: Annotated[aioredis.Redis, Depends(get_redis_dep)],
) -> RedisQueue:
    return RedisQueue(redis)


async def get_state_store(
    redis: Annotated[aioredis.Redis, Depends(get_redis_dep)],
) -> StateStore:
    return StateStore(redis)


# Type aliases for cleaner route signatures
RedisDep = Annotated[aioredis.Redis, Depends(get_redis_dep)]
QueueDep = Annotated[RedisQueue, Depends(get_queue)]
StateStoreDep = Annotated[StateStore, Depends(get_state_store)]
