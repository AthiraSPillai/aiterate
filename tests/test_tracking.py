from aiterate.tracking.mlflow_tracker import MLflowTracker


class FailingMLflow:
    def set_experiment(self, name):
        raise RuntimeError(f"cannot connect to {name}")

    def start_run(self, run_name):
        raise AssertionError("start_run should not be called after set_experiment fails")


def test_mlflow_tracker_disables_itself_when_server_is_unreachable():
    tracker = MLflowTracker.__new__(MLflowTracker)
    tracker.mlflow = FailingMLflow()
    tracker.enabled = True
    tracker.last_error = None
    tracker.tracking_uri = None

    tracker.start_run("demo", {"artifact_id": "art_123"})

    assert tracker.enabled is False
    assert "MLflow start run failed" in tracker.last_error


class TrackingUriMLflow:
    def set_experiment(self, name):
        raise AssertionError("MLflow experiment API should not be called when preflight fails")

    def start_run(self, run_name):
        raise AssertionError("MLflow run API should not be called when preflight fails")


def test_mlflow_tracker_skips_quickly_when_http_server_is_unavailable(monkeypatch):
    def fail_fast(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("requests.get", fail_fast)
    tracker = MLflowTracker.__new__(MLflowTracker)
    tracker.mlflow = TrackingUriMLflow()
    tracker.enabled = True
    tracker.last_error = None
    tracker.tracking_uri = "http://mlflow:5000"

    tracker.start_run("demo", {"artifact_id": "art_123"})

    assert tracker.enabled is False
    assert "Optimization continued without tracking" in tracker.last_error
