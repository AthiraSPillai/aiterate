from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select

from aiterate.db import JobRecord, session_scope
from aiterate.domain import OptimizationRequest, new_id
from aiterate.optimizer import SkillOptInspiredOptimizer
from aiterate.run_store import RunStore


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobEnvelope(BaseModel):
    id: str
    kind: str
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    created_at: datetime
    updated_at: datetime


class JobStore:
    def enqueue(self, kind: str, payload: dict[str, Any]) -> JobEnvelope:
        now = datetime.now(UTC)
        record = JobRecord(
            id=new_id("job"),
            kind=kind,
            status=JobStatus.QUEUED.value,
            payload=payload,
            attempts=0,
            created_at=now,
            updated_at=now,
        )
        with session_scope() as session:
            session.add(record)
        return self.get(record.id)

    def get(self, job_id: str) -> JobEnvelope:
        with session_scope() as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                raise KeyError(job_id)
            return _to_envelope(record)

    def claim_next(self, kind: str = "optimization") -> JobEnvelope | None:
        with session_scope() as session:
            record = session.scalars(
                select(JobRecord)
                .where(JobRecord.kind == kind, JobRecord.status == JobStatus.QUEUED.value)
                .order_by(JobRecord.created_at)
                .limit(1)
            ).first()
            if record is None:
                return None
            record.status = JobStatus.RUNNING.value
            record.attempts += 1
            record.updated_at = datetime.now(UTC)
            return _to_envelope(record)

    def complete(self, job_id: str, result: dict[str, Any]) -> JobEnvelope:
        return self._finish(job_id, JobStatus.SUCCEEDED, result=result)

    def fail(self, job_id: str, error: str) -> JobEnvelope:
        return self._finish(job_id, JobStatus.FAILED, error=error)

    def _finish(
        self,
        job_id: str,
        status: JobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> JobEnvelope:
        with session_scope() as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                raise KeyError(job_id)
            record.status = status.value
            record.result = result
            record.error = error
            record.updated_at = datetime.now(UTC)
            return _to_envelope(record)


def run_one_optimization_job() -> JobEnvelope | None:
    store = JobStore()
    job = store.claim_next("optimization")
    if job is None:
        return None
    try:
        request = OptimizationRequest.model_validate(job.payload)
        run = SkillOptInspiredOptimizer().optimize(request)
        RunStore().append(run)
        return store.complete(job.id, run.model_dump(mode="json"))
    except Exception as exc:
        return store.fail(job.id, str(exc))


def _to_envelope(record: JobRecord) -> JobEnvelope:
    return JobEnvelope(
        id=record.id,
        kind=record.kind,
        status=JobStatus(record.status),
        payload=record.payload,
        result=record.result,
        error=record.error,
        attempts=record.attempts,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
