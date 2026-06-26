from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aiterate.config import settings
from aiterate.tracking.base import Tracker


class MLflowTracker(Tracker):
    def __init__(self, tracking_uri: str | None = None, token: str | None = None) -> None:
        self.enabled = True
        self.last_error: str | None = None
        try:
            import mlflow
        except ImportError:
            self.mlflow = None
            self.enabled = False
            self.last_error = "MLflow package is not installed."
            return
        self.mlflow = mlflow
        token = token or settings.mlflow_tracking_token
        if token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = token
        uri = tracking_uri or settings.mlflow_tracking_uri
        if uri:
            self.mlflow.set_tracking_uri(uri)

    def start_run(self, name: str, metadata: dict[str, Any]) -> None:
        if not self.mlflow or not self.enabled:
            return
        if not self._safe("start run", lambda: self.mlflow.set_experiment("aiterate")):
            return
        if not self._safe("start run", lambda: self.mlflow.start_run(run_name=name)):
            return
        for key, value in metadata.items():
            self.log_param(key, str(value))

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self.mlflow and self.enabled:
            self._safe("log metric", lambda: self.mlflow.log_metric(key, value, step=step))

    def log_param(self, key: str, value: str) -> None:
        if self.mlflow and self.enabled:
            self._safe("log param", lambda: self.mlflow.log_param(key, value[:250]))

    def log_artifact(self, path: Path) -> None:
        if self.mlflow and self.enabled:
            self._safe("log artifact", lambda: self.mlflow.log_artifact(str(path)))

    def end_run(self) -> None:
        if self.mlflow and self.enabled:
            self._safe("end run", self.mlflow.end_run)

    def _safe(self, action: str, callback) -> bool:
        try:
            callback()
        except Exception as exc:  # MLflow raises provider-specific network/auth exceptions.
            self.enabled = False
            self.last_error = f"MLflow {action} failed: {exc}"
            return False
        return True
