from __future__ import annotations

from datetime import UTC, datetime

from aiterate.db import ProjectSettingsRecord, session_scope
from aiterate.domain import ProjectSettings


class ProjectSettingsStore:
    def get(self, project_name: str) -> ProjectSettings:
        with session_scope() as session:
            record = session.get(ProjectSettingsRecord, project_name)
            if record is None:
                return ProjectSettings(project_name=project_name)
            return ProjectSettings.model_validate(record.payload)

    def upsert(self, settings: ProjectSettings) -> ProjectSettings:
        settings = settings.model_copy(update={"project_name": settings.project_name.strip()})
        with session_scope() as session:
            session.merge(
                ProjectSettingsRecord(
                    project_name=settings.project_name,
                    payload=settings.model_dump(mode="json"),
                    updated_at=datetime.now(UTC),
                )
            )
        return settings
