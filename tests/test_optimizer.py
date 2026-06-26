from aiterate.config import settings
from aiterate.domain import OptimizationRequest, PriorityRule, ProviderConfig, ProviderKind
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
    gated_versions = [*run.accepted_versions[1:], *run.rejected_versions]
    assert all("gate_rule" in version.metadata for version in gated_versions)
    assert all("gate_reason" in version.metadata for version in gated_versions)
    assert all("score_delta" in version.metadata for version in gated_versions)


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


def test_optimizer_records_train_validation_split():
    request = OptimizationRequest(
        name="split-demo",
        raw_data="\n\n".join(
            [
                "Case 1: cite the billing policy.",
                "Case 2: escalate missing account data.",
                "Case 3: refuse unsupported claims.",
                "Case 4: keep tone concise.",
            ]
        ),
        policies=[PriorityRule(id="cite", text="Always cite sources.", weight=1)],
        iterations=1,
        validation_split=0.5,
        seed=3,
    )

    run = SkillOptInspiredOptimizer().optimize(request)

    assert run.optimizer["validation_split"] == 0.5
    assert run.optimizer["train_case_count"] == 2
    assert run.optimizer["validation_case_count"] == 2
    assert run.accepted_versions[0].metadata["scored_on"] == "validation_holdout"


def test_optimizer_separates_data_policy_and_knowledge_context():
    request = OptimizationRequest(
        name="context-roles-demo",
        raw_data="Case 1: answer billing question.\n\nCase 2: escalate contradictory account data.",
        policy_context="Must cite source documents.\nEscalate contradictory account state.",
        knowledge_base_context="KB article: prorated billing changes should cite invoice line items.",
        iterations=1,
    )

    run = SkillOptInspiredOptimizer().optimize(request)

    assert any(rule.id.startswith("policy_file_") for rule in run.policy_set.rules)
    assert run.optimizer["context_roles"]["data_examples"] == "train_test_cases"
    assert run.optimizer["policy_context_hash"]
    assert run.optimizer["knowledge_base_hash"]
    assert run.accepted_versions[0].metadata["context_roles"]["knowledge_base_context"] == "grounding_references"


def test_optimizer_does_not_duplicate_uploaded_policy_text():
    request = OptimizationRequest(
        name="context-roles-demo",
        raw_data="Case 1: answer billing question.",
        policies=[PriorityRule(id="cite_sources", text="Must cite source documents.", weight=1)],
        policy_context="Must cite source documents.\nEscalate contradictory account state.",
        iterations=1,
    )

    run = SkillOptInspiredOptimizer().optimize(request)

    policy_texts = [" ".join(rule.text.lower().split()) for rule in run.policy_set.rules]
    assert policy_texts.count("must cite source documents.") == 1
    assert any(rule.id.startswith("policy_file_") for rule in run.policy_set.rules)


def test_optimizer_stops_when_budget_cap_is_reached():
    request = OptimizationRequest(
        name="budget-demo",
        raw_data="Support answers must cite sources and escalate incomplete cases.",
        policies=[PriorityRule(id="cite", text="Always cite sources.", weight=1)],
        iterations=5,
        max_budget_usd=0,
    )

    run = SkillOptInspiredOptimizer().optimize(request)

    assert run.optimizer["max_budget_usd"] == 0
    assert "budget_stop_reason" in run.optimizer
    assert len(run.accepted_versions) == 1


def test_optimizer_can_validate_best_version_with_target_model():
    request = OptimizationRequest(
        name="target-validation-demo",
        raw_data="Support answers must cite sources and escalate incomplete cases.",
        policies=[PriorityRule(id="cite", text="Always cite sources.", weight=1)],
        provider=ProviderConfig(kind=ProviderKind.MOCK, model="mock-optimizer"),
        target_provider=ProviderConfig(kind=ProviderKind.MOCK, model="mock-target"),
        run_target_validation=True,
        iterations=1,
    )

    run = SkillOptInspiredOptimizer().optimize(request)

    assert run.optimizer["run_target_validation"] is True
    assert run.behavior_report is not None
    assert run.behavior_report.case_count >= 1
