"""
tests/test_integration.py
--------------------------
Integration tests for the production layer.

These tests:
  - Mock graph.ainvoke so they don't need a live LLM / Redis.
  - Test the full API → queue → worker → SSE pipeline using
    an in-process Redis mock (fakeredis).

Run:
    cd my_agent/production
    pip install pytest pytest-asyncio fakeredis httpx
    pytest tests/test_integration.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient

# Ensure production root is on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fake_redis():
    """In-process fake Redis with Pub/Sub support."""
    r = FakeRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def api_client(fake_redis):
    """FastAPI test client wired to fake Redis."""
    # Patch get_redis so the app uses fake Redis
    with patch("core.redis_client._redis_pool", fake_redis):
        from api.main import create_app

        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fake_agent_state(user_message: str) -> dict[str, Any]:
    """Return a minimal AgentState-like dict (no real LLM needed)."""
    from langchain_core.messages import AIMessage, HumanMessage

    return {
        "messages": [
            HumanMessage(content=user_message),
            AIMessage(content=f"Echo: {user_message}"),
        ],
        "final_answer": f"Echo: {user_message}",
        "next": None,
        "current_agent": None,
        "error": None,
        "step_count": 1,
        "scratchpad": "",
        "tool_calls": [],
        "retrieved_context": [],
        "metadata": {},
        "artifacts": [],
        "system_instruction": "",
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_job_returns_202(api_client):
    """POST /jobs must return 202 with job_id and stream_url."""
    resp = await api_client.post(
        "/jobs",
        json={"session_id": "sess-1", "user_message": "Hello"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert "/stream" in body["stream_url"]
    assert "ws://" in body["ws_url"]


@pytest.mark.asyncio
async def test_get_job_initial_status(api_client):
    """GET /jobs/{job_id} should reflect queued status immediately after POST."""
    post = await api_client.post(
        "/jobs",
        json={"session_id": "sess-2", "user_message": "Test"},
    )
    job_id = post.json()["job_id"]

    get = await api_client.get(f"/jobs/{job_id}")
    assert get.status_code == 200
    assert get.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_get_job_not_found(api_client):
    """GET /jobs/{unknown_id} → 404."""
    resp = await api_client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_execute_job_end_to_end(fake_redis):
    """
    Unit test for execute_job:
    - Patches graph.ainvoke to return a fake state.
    - Verifies the job result is saved as 'done'.
    - Verifies the Pub/Sub event is published.
    """
    from models.job import JobRequest, JobResult
    from services.queue import RedisQueue
    from services.state_store import StateStore
    from worker.runner import execute_job

    state_store = StateStore(fake_redis)
    queue = RedisQueue(fake_redis)

    # Pre-save a queued job result (normally done by POST /jobs)
    job_result = JobResult(session_id="sess-3", user_message="Hello worker")
    job_id = job_result.job_id
    await state_store.save_job_result(job_result)

    payload = {
        "job_id": job_id,
        "session_id": "sess-3",
        "user_message": "Hello worker",
        "metadata": {},
    }

    fake_state = _fake_agent_state("Hello worker")

    # Subscribe to pub/sub before executing
    pubsub = fake_redis.pubsub()
    channel = f"job:{job_id}:events"
    await pubsub.subscribe(channel)

    with patch("worker.runner._invoke_graph", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = fake_state
        result = await execute_job(payload, fake_redis, state_store)

    assert result.status == "done"
    assert result.final_answer == "Echo: Hello worker"

    # Verify done event published
    events = []
    async for msg in pubsub.listen():
        if msg["type"] == "message":
            events.append(json.loads(msg["data"]))
            break
    assert any(e.get("event") == "done" for e in events)

    await pubsub.unsubscribe(channel)
    await pubsub.aclose()


@pytest.mark.asyncio
async def test_session_state_persisted(fake_redis):
    """Second job with same session_id should see prior messages in state."""
    from langchain_core.messages import HumanMessage

    from models.job import JobResult
    from services.state_store import StateStore
    from worker.runner import execute_job

    store = StateStore(fake_redis)
    session_id = "sess-persist"

    def _make_state(msgs):
        from langchain_core.messages import AIMessage

        return {
            **_fake_agent_state("turn2"),
            "messages": msgs + [AIMessage(content="Reply")],
        }

    # First job
    result1 = JobResult(session_id=session_id, user_message="Turn 1")
    await store.save_job_result(result1)
    payload1 = {
        "job_id": result1.job_id,
        "session_id": session_id,
        "user_message": "Turn 1",
        "metadata": {},
    }
    with patch("worker.runner._invoke_graph", new_callable=AsyncMock) as mock:
        mock.return_value = _fake_agent_state("Turn 1")
        await execute_job(payload1, fake_redis, store)

    # Second job — state should include Turn 1 messages
    result2 = JobResult(session_id=session_id, user_message="Turn 2")
    await store.save_job_result(result2)
    payload2 = {
        "job_id": result2.job_id,
        "session_id": session_id,
        "user_message": "Turn 2",
        "metadata": {},
    }

    captured: list[dict] = []

    async def _capture_state(state):
        captured.append(state)
        return _fake_agent_state("Turn 2")

    with patch("worker.runner._invoke_graph", new_callable=AsyncMock) as mock:
        mock.side_effect = _capture_state
        await execute_job(payload2, fake_redis, store)

    # The state passed to the second invocation must contain Turn 1's messages
    assert len(captured) == 1
    messages = captured[0]["messages"]
    contents = [m.content for m in messages]
    assert "Turn 1" in contents or any("Turn 1" in c for c in contents)


@pytest.mark.asyncio
async def test_health_endpoint(api_client):
    resp = await api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
