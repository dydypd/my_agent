"""
api/routes/jobs.py
------------------
REST endpoints for job submission and result polling.

POST /jobs          → submit a job (returns immediately, 202)
GET  /jobs/{job_id} → poll current result / status
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from api.dependencies import QueueDep, StateStoreDep
from models.job import JobRequest, JobResult, JobSubmitResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobSubmitResponse,
    summary="Submit a new agent job",
)
async def submit_job(
    body: JobRequest,
    request: Request,
    queue: QueueDep,
    store: StateStoreDep,
) -> JobSubmitResponse:
    """
    Accept a job request, persist an initial JobResult with status=queued,
    push the job to the Redis queue, and return immediately (non-blocking).
    """
    # Build initial result record
    job_result = JobResult(
        session_id=body.session_id,
        user_message=body.user_message,
        metadata=body.metadata,
    )
    job_id = job_result.job_id

    # Persist queued state so GET /jobs/{job_id} works immediately
    await store.save_job_result(job_result)

    # Push to queue (producer)
    await queue.push_job(body, job_id)
    logger.info("Job %s queued for session %s", job_id, body.session_id)

    base = str(request.base_url).rstrip("/")
    return JobSubmitResponse(
        job_id=job_id,
        session_id=body.session_id,
        stream_url=f"{base}/jobs/{job_id}/stream",
        ws_url=f"{base.replace('http', 'ws')}/ws/{body.session_id}",
    )


@router.get(
    "/{job_id}",
    response_model=JobResult,
    summary="Get the current status / result of a job",
)
async def get_job(job_id: str, store: StateStoreDep) -> JobResult:
    """
    Return the current JobResult from Redis.
    The client can poll this endpoint or use the stream/WebSocket equivalent.
    """
    result = await store.load_job_result(job_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return result
