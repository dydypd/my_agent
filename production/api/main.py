"""
api/main.py
-----------
FastAPI application factory.

Lifespan
--------
- On startup: initialise Redis pool, Recover queue orphans (via queue service).
- On shutdown: close Redis pool cleanly.

All routes are registered here and receive shared dependencies via DI.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure production/ root is on sys.path when invoked directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.routes.jobs import router as jobs_router
from api.routes.stream import router_stream
from api.routes.ws import router_ws
from core.config import get_settings
from core.redis_client import close_redis, get_redis
from services.queue import RedisQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle."""
    logger.info("API starting up…")
    redis = await get_redis()
    # Recover any jobs orphaned by a worker crash
    queue = RedisQueue(redis)
    recovered = await queue.requeue_orphaned()
    if recovered:
        logger.info("Recovered %d orphaned job(s) at startup", recovered)
    yield
    logger.info("API shutting down…")
    await close_redis()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Multi-Agent API Gateway",
        description=(
            "Production API gateway for the LangGraph Multi-Agent system. "
            "Submit jobs via REST, stream results via SSE or WebSocket."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — tighten allowed_origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(jobs_router)
    app.include_router(router_stream)
    app.include_router(router_ws)

    @app.get("/health", tags=["meta"], summary="Health check")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )
