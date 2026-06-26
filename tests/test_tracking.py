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

    tracker.start_run("demo", {"artifact_id": "art_123"})

    assert tracker.enabled is False
    assert "MLflow start run failed" in tracker.last_error
