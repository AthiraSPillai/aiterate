from aiterate.config import settings
from aiterate.domain import OptimizationRequest, PriorityRule
from aiterate.optimizer import SkillOptInspiredOptimizer


def test_optimizer_accepts_versions_without_local_git_by_default():
    request = OptimizationRequest(
        name="demo",
        raw_data="Answer support questions with citations. Escalate incomplete cases.",
        policies=[
            PriorityRule(id="cite", text="Always cite sources.", weight=0.5),
            PriorityRule(id="escalate", text="Escalate incomplete data.", weight=0.5),
        ],
        iterations=2,
    )
    assert settings.enable_local_git is False
    optimizer = SkillOptInspiredOptimizer()
    run = optimizer.optimize(request)
    assert run.best_version is not None
    assert run.best_version.score > 0
    assert run.accepted_versions
    assert run.optimizer["framework"] == "skillopt"
    assert run.accepted_versions[0].metadata["skillopt_stage"] == "baseline"
    assert all("skillopt_gate_action" in version.metadata for version in run.accepted_versions[1:])


def test_optimizer_uses_user_baseline_as_skillopt_start_state():
    baseline = "Answer only from the supplied support policy."
    request = OptimizationRequest(
        name="baseline-demo",
        raw_data="Support answers must cite sources and escalate incomplete cases.",
        baseline_artifact=baseline,
        policies=[
            PriorityRule(id="cite", text="Always cite sources.", weight=0.5),
            PriorityRule(id="escalate", text="Escalate incomplete data.", weight=0.5),
        ],
        iterations=1,
    )

    run = SkillOptInspiredOptimizer().optimize(request)

    assert run.accepted_versions[0].content == baseline
    assert run.accepted_versions[0].change_summary.startswith("User baseline")
    assert run.accepted_versions[0].metadata["baseline_source"] == "user"
    assert run.rejected_versions or len(run.accepted_versions) > 1
