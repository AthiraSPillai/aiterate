from aiterate.domain import TrackerKind
from aiterate.tracking.base import NoopTracker, Tracker
from aiterate.tracking.langsmith_tracker import LangSmithTracker
from aiterate.tracking.mlflow_tracker import MLflowTracker


def build_tracker(kind: TrackerKind) -> Tracker:
    if kind == TrackerKind.MLFLOW:
        return MLflowTracker()
    if kind == TrackerKind.LANGSMITH:
        return LangSmithTracker()
    return NoopTracker()

