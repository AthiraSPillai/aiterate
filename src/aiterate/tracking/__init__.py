from aiterate.domain import TrackerKind
from aiterate.tracking.base import NoopTracker, Tracker
from aiterate.tracking.langsmith_tracker import LangSmithTracker
from aiterate.tracking.mlflow_tracker import MLflowTracker


def build_tracker(
    kind: TrackerKind,
    tracker_uri: str | None = None,
    api_key: str | None = None,
    project: str | None = None,
) -> Tracker:
    if kind == TrackerKind.MLFLOW:
        return MLflowTracker(tracking_uri=tracker_uri, token=api_key)
    if kind == TrackerKind.LANGSMITH:
        return LangSmithTracker(endpoint_url=tracker_uri, api_key=api_key, project=project)
    return NoopTracker()
