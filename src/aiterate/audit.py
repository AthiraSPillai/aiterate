from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aiterate.auth import CurrentUser
from aiterate.db import AuditLogRecord, session_scope
from aiterate.domain import new_id


class AuditLogger:
    def log(
        self,
        action: str,
        actor: CurrentUser,
        target_type: str,
        target_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with session_scope() as session:
            session.add(
                AuditLogRecord(
                    id=new_id("audit"),
                    actor_id=actor.id,
                    actor_role=actor.role.value,
                    action=action,
                    target_type=target_type,
                    target_id=target_id,
                    payload=payload or {},
                    created_at=datetime.now(UTC),
                )
            )
