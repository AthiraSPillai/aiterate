from __future__ import annotations

from sqlalchemy import select

from aiterate.db import RunRecord, session_scope
from aiterate.domain import OptimizationRun


class RunStore:
    def append(self, run: OptimizationRun) -> None:
        with session_scope() as session:
            session.merge(
                RunRecord(
                    id=run.id,
                    name=run.name,
                    artifact_id=run.artifact_id,
                    payload=run.model_dump(mode="json"),
                    created_at=run.created_at,
                )
            )

    def list_grouped(self) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}
        with session_scope() as session:
            records = session.scalars(select(RunRecord).order_by(RunRecord.created_at.desc())).all()
        for record in records:
            run = record.payload
            approval = run.get("approval") or {}
            grouped.setdefault(run["name"], []).append(
                {
                    "id": run["id"],
                    "artifact_id": run["artifact_id"],
                    "created_at": run["created_at"],
                    "best_score": (run.get("best_version") or {}).get("score"),
                    "accepted_versions": len(run.get("accepted_versions") or []),
                    "rejected_versions": len(run.get("rejected_versions") or []),
                    "provider": run.get("provider", {}).get("kind"),
                    "model": run.get("provider", {}).get("model"),
                    "estimated_cost": (run.get("cost_estimate") or {}).get("total_cost"),
                    "estimated_cost_currency": (run.get("cost_estimate") or {}).get("currency"),
                    "approval": approval,
                    "approved": approval.get("status") == "approved",
                    "approved_version_id": approval.get("version_id"),
                }
            )
        return dict(sorted(grouped.items()))

    def get(self, run_id: str) -> OptimizationRun:
        with session_scope() as session:
            record = session.get(RunRecord, run_id)
            if record is None:
                raise KeyError(run_id)
            return OptimizationRun.model_validate(record.payload)

    def delete(self, run_id: str) -> OptimizationRun:
        with session_scope() as session:
            record = session.get(RunRecord, run_id)
            if record is None:
                raise KeyError(run_id)
            run = OptimizationRun.model_validate(record.payload)
            session.delete(record)
            return run

    def delete_by_name(self, name: str) -> list[OptimizationRun]:
        with session_scope() as session:
            records = session.scalars(select(RunRecord).where(RunRecord.name == name)).all()
            if not records:
                raise KeyError(name)
            runs = [OptimizationRun.model_validate(record.payload) for record in records]
            for record in records:
                session.delete(record)
            return runs

    def approve(self, run_id: str, approval: dict) -> dict:
        with session_scope() as session:
            record = session.get(RunRecord, run_id)
            if record is None:
                raise KeyError(run_id)
            payload = dict(record.payload)
            payload["approval"] = approval
            record.payload = payload
        return approval
