from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any


class Tracker(ABC):
    def start_run(self, name: str, metadata: dict[str, Any]) -> None:
        pass

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        pass

    def log_param(self, key: str, value: str) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass

    def end_run(self) -> None:
        pass


class NoopTracker(Tracker):
    pass

