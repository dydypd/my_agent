"""
worker/retry.py
---------------
Configurable async retry with exponential back-off.

Usage
-----
    result = await run_with_retry(my_coroutine_fn, arg1, arg2, settings=settings)

The decorated coroutine is retried up to `MAX_RETRIES` times on any of the
listed transient exceptions.  After exhausting retries the last exception is
re-raised.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

from core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exception types considered transient (safe to retry)
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
    OSError,
)

# Also retry generic Exception so LLM / tool failures are covered.
# The final failure is always re-raised to the caller.


async def run_with_retry(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    attempt_callback: Callable[[int, Exception], None] | None = None,
    **kwargs: Any,
) -> tuple[T, int]:
    """
    Call `fn(*args, **kwargs)` up to MAX_RETRIES times.

    Returns
    -------
    (result, attempts_used)
        attempts_used is 0-indexed (0 = succeeded on first try).

    Raises
    ------
    The last exception if all retries are exhausted.
    """
    settings = get_settings()
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(settings.max_retries + 1):
        try:
            result = await asyncio.wait_for(
                fn(*args, **kwargs),
                timeout=settings.graph_invoke_timeout,
            )
            return result, attempt
        except Exception as exc:
            last_exc = exc
            if attempt < settings.max_retries:
                delay = settings.retry_backoff_base ** attempt
                logger.warning(
                    "Attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1,
                    settings.max_retries + 1,
                    exc,
                    delay,
                )
                if attempt_callback:
                    attempt_callback(attempt, exc)
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "All %d attempts exhausted. Last error: %s",
                    settings.max_retries + 1,
                    exc,
                )

    raise last_exc
