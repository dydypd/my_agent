"""
models/job.py
-------------
Shared Pydantic v2 schemas for job lifecycle.
Both the API and the Worker import these — they live here to avoid circular deps.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobRequest(BaseModel):
    """Payload the API client sends when submitting a job."""

    session_id: str = Field(
        description="Ties this job to a persisted conversation state. "
        "Multiple jobs with the same session_id will share message history."
    )
    user_message: str = Field(
        description="The human turn to inject into the graph."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra context forwarded to AgentState.metadata.",
    )


class JobResult(BaseModel):
    """Stored in Redis and returned by GET /jobs/{job_id}."""

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    status: Literal["queued", "running", "done", "failed"] = "queued"

    final_answer: str | None = None
    error: str | None = None
    retry_count: int = 0

    created_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None

    # Echo back the original request for traceability
    user_message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_running(self) -> "JobResult":
        return self.model_copy(update={"status": "running"})

    def mark_done(self, final_answer: str) -> "JobResult":
        return self.model_copy(
            update={
                "status": "done",
                "final_answer": final_answer,
                "completed_at": _now(),
            }
        )

    def mark_failed(self, error: str, retry_count: int) -> "JobResult":
        return self.model_copy(
            update={
                "status": "failed",
                "error": error,
                "retry_count": retry_count,
                "completed_at": _now(),
            }
        )


class JobSubmitResponse(BaseModel):
    """Returned immediately by POST /jobs."""

    job_id: str
    session_id: str
    status: Literal["queued"] = "queued"
    stream_url: str
    ws_url: str
