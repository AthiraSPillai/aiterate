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
                }
            )
        return dict(sorted(grouped.items()))
