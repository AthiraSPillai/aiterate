from __future__ import annotations

from pathlib import Path
from typing import Any

from aiterate.config import settings
from aiterate.tracking.base import Tracker


class MLflowTracker(Tracker):
    def __init__(self) -> None:
        try:
            import mlflow
        except ImportError:
            self.mlflow = None
            return
        self.mlflow = mlflow
        if settings.mlflow_tracking_uri:
            self.mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def start_run(self, name: str, metadata: dict[str, Any]) -> None:
        if not self.mlflow:
            return
        self.mlflow.set_experiment("aiterate")
        self.mlflow.start_run(run_name=name)
        for key, value in metadata.items():
            self.log_param(key, str(value))

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self.mlflow:
            self.mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: str) -> None:
        if self.mlflow:
            self.mlflow.log_param(key, value[:250])

    def log_artifact(self, path: Path) -> None:
        if self.mlflow:
            self.mlflow.log_artifact(str(path))

    def end_run(self) -> None:
        if self.mlflow:
            self.mlflow.end_run()

