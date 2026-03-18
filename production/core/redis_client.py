"""
core/redis_client.py
--------------------
Manages a shared async Redis connection pool.

Usage
-----
    from core.redis_client import get_redis
    redis = await get_redis()
    await redis.set("key", "value")

Call `close_redis()` on application shutdown to flush the pool.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from core.config import get_settings

_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return a lazily-initialised async Redis client (connection pool)."""
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )
    return _redis_pool


async def close_redis() -> None:
    """Close the connection pool — call on application shutdown."""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None
