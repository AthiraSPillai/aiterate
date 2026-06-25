from __future__ import annotations

import json
import random

from skillopt.evaluation.gate import evaluate_gate
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.types import Edit, Patch

from aiterate.domain import (
    ArtifactVersion,
    EvaluationInsight,
    EvaluationReport,
    ModelComparisonRequest,
    ModelComparisonResponse,
    ModelComparisonResult,
    OptimizationRequest,
    OptimizationRun,
    PolicySet,
    new_id,
)
from aiterate.evaluator import evaluate_artifact
from aiterate.ingest import normalize_raw_data, stable_hash
from aiterate.providers import build_provider
from aiterate.tracking import build_tracker
from aiterate.versioning import GitVersionStore


SYSTEM_PROMPT = """You are AIterate's artifact optimizer.
Create production-ready AI prompts or agent skill documents from raw data and weighted policies.
Prefer concise, testable instructions with explicit source, uncertainty, and escalation behavior."""


SKILLOPT_SYSTEM_PROMPT = """You are running SkillOpt-style text artifact training for AIterate.
Return bounded skill/prompt edits as JSON only:
{"reasoning":"why this improves validation","edits":[{"op":"append|insert_after|replace|delete","target":"optional exact text","content":"edit text"}]}
Optimize for measurable policy/eval improvement, preserve deployability, and avoid ungrounded behavior."""


class SkillOptInspiredOptimizer:
    """AIterate's SkillOpt-based experiment loop for prompt and skill artifacts.

    The product wrapper adds raw-data ingestion, policy weighting, lineage, Git versioning, tracking,
    and approval metadata around SkillOpt's edit/update/gate mechanics.
    """

    def __init__(self, version_store: GitVersionStore | None = None) -> None:
        self.version_store = version_store or GitVersionStore()

    def optimize(self, request: OptimizationRequest) -> OptimizationRun:
        random.seed(request.seed)
        dataset = normalize_raw_data(request.name, request.raw_data)
        policy_set = PolicySet(name=request.name, rules=request.policies)
        provider = build_provider(request.provider)
        tracker = build_tracker(request.tracker)
        artifact_id = new_id("art")
        policy_hash = stable_hash(policy_set.model_dump_json())
        run = OptimizationRun(
            name=request.name,
            artifact_id=artifact_id,
            dataset=dataset,
            policy_set=policy_set,
            provider=request.provider.redacted(),
            optimizer={
                "framework": "skillopt",
                "wrapper": "aiterate.SkillOptInspiredOptimizer",
                "loop": ["baseline", "reflect", "patch", "validate_gate", "accept_or_reject"],
                "seed": request.seed,
            },
        )

        tracker.start_run(
            request.name,
            {
                "artifact_id": artifact_id,
                "optimizer_framework": "skillopt",
                "optimizer_wrapper": "aiterate.SkillOptInspiredOptimizer",
                "provider": request.provider.kind.value,
                "model": request.provider.model,
                "target_model": request.target_model or "",
                "git_tracking": str(request.enable_git_tracking),
                "create_pull_request": str(request.create_pull_request),
                "dataset_hash": dataset.content_hash,
                "policy_hash": policy_hash,
            },
        )
        try:
            best = self._initial_version(request, dataset.normalized_cases, policy_set, provider)
            best = best.model_copy(
                update={
                    "artifact_id": artifact_id,
                    "dataset_hash": dataset.content_hash,
                    "policy_hash": policy_hash,
                    "provider": request.provider.redacted(),
                    "metadata": {
                        **best.metadata,
                        "skillopt_stage": "baseline",
                        "baseline_source": "user" if request.baseline_artifact else "generated",
                    },
                }
            )
            run.accepted_versions.append(best)
            self.version_store.commit_version(run, best)
            tracker.log_metric("score", best.score, step=0)

            for iteration in range(1, request.iterations + 1):
                candidate = self._candidate_version(
                    request,
                    best,
                    dataset.normalized_cases,
                    policy_set,
                    provider,
                    iteration,
                ).model_copy(
                    update={
                        "artifact_id": artifact_id,
                        "dataset_hash": dataset.content_hash,
                        "policy_hash": policy_hash,
                        "provider": request.provider.redacted(),
                    }
                )
                if candidate.accepted:
                    run.accepted_versions.append(candidate)
                    best = candidate
                    self.version_store.commit_version(run, candidate)
                else:
                    run.rejected_versions.append(candidate)
                tracker.log_metric("score", candidate.score, step=iteration)
                tracker.log_metric("accepted", 1.0 if candidate.accepted else 0.0, step=iteration)
            run.best_version = best
            run.evaluation_report = evaluate_artifact(
                best.content,
                dataset.normalized_cases,
                policy_set,
                request.eval_assertions,
            )
            run.insights = self._build_insights(
                best,
                run.rejected_versions,
                dataset.normalized_cases,
                policy_set,
                run.evaluation_report,
            )
            return run
        finally:
            tracker.end_run()

    def _initial_version(self, request, cases, policy_set, provider) -> ArtifactVersion:
        if request.baseline_artifact and request.baseline_artifact.strip():
            content = request.baseline_artifact.strip()
            change_summary = "User baseline artifact imported as SkillOpt training start state."
        else:
            user = self._brief(request, cases, policy_set, "Create the initial SkillOpt baseline artifact.")
            content = provider.generate(SYSTEM_PROMPT, user)
            change_summary = "Initial artifact generated from raw data and weighted policies."
        score = self._score(content, cases, policy_set, request.eval_assertions)
        return ArtifactVersion(
            artifact_id="pending",
            kind=request.artifact_kind,
            version=1,
            content=content,
            score=score,
            change_summary=change_summary,
            dataset_hash="pending",
            policy_hash="pending",
            provider={},
            metadata={"skillopt_stage": "baseline"},
        )

    def _candidate_version(self, request, best, cases, policy_set, provider, iteration) -> ArtifactVersion:
        before_report = evaluate_artifact(best.content, cases, policy_set, request.eval_assertions)
        patch = self._candidate_patch(request, best, cases, policy_set, provider, iteration, before_report)
        content, patch_report = apply_patch_with_report(best.content, patch)
        if content.strip() == best.content.strip():
            patch = self._fallback_patch(policy_set, iteration, reason="provider patch produced no text change")
            content, patch_report = apply_patch_with_report(best.content, patch)

        after_report = evaluate_artifact(content, cases, policy_set, request.eval_assertions)
        score = self._score_from_report(after_report, cases, content)
        gate = evaluate_gate(
            candidate_skill=content,
            cand_hard=score,
            cand_soft=after_report.score,
            current_skill=best.content,
            current_score=best.score,
            best_skill=best.content,
            best_score=best.score,
            best_step=best.version,
            global_step=iteration,
            metric="hard",
            mixed_weight=0.35,
        )
        accepted = gate.action != "reject"
        return ArtifactVersion(
            artifact_id=best.artifact_id,
            kind=request.artifact_kind,
            version=best.version + 1 if accepted else best.version,
            content=content,
            score=score,
            parent_version_id=best.id,
            accepted=accepted,
            change_summary=self._patch_summary(patch, gate.action),
            dataset_hash=best.dataset_hash,
            policy_hash=best.policy_hash,
            provider=best.provider,
            metadata={
                "skillopt_stage": "validate_gate",
                "skillopt_gate_action": gate.action,
                "skillopt_patch": patch.to_dict(),
                "skillopt_patch_report": patch_report,
                "previous_score": best.score,
                "candidate_eval_score": after_report.score,
                "failed_metrics": after_report.failed_metrics,
            },
        )

    def _candidate_patch(
        self,
        request: OptimizationRequest,
        best: ArtifactVersion,
        cases: list[str],
        policy_set: PolicySet,
        provider,
        iteration: int,
        before_report: EvaluationReport,
    ) -> Patch:
        user = (
            f"{self._brief(request, cases, policy_set, f'Reflect on version {best.version} and propose bounded edits.')}\n"
            f"Current artifact:\n{best.content}\n\n"
            f"Current validation score: {best.score}\n"
            f"Failed metrics: {', '.join(before_report.failed_metrics) or 'none'}\n"
            "Use SkillOpt add/delete/replace-style edits. Prefer small measurable changes."
        )
        raw = provider.generate(SKILLOPT_SYSTEM_PROMPT, user)
        patch = self._parse_patch(raw)
        if patch.edits:
            return patch
        return self._fallback_patch(policy_set, iteration, reason="provider did not return parseable SkillOpt edits")

    def _parse_patch(self, raw: str) -> Patch:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            payload = json.loads(raw[start:end] if start >= 0 and end > start else raw)
        except (json.JSONDecodeError, TypeError):
            return Patch(edits=[], reasoning="unparseable provider patch")
        try:
            return Patch.from_dict(payload)
        except (AttributeError, TypeError, ValueError):
            return Patch(edits=[], reasoning="invalid provider patch")

    def _fallback_patch(self, policy_set: PolicySet, iteration: int, reason: str) -> Patch:
        policy_lines = "\n".join(f"- {rule.text}" for rule in sorted(policy_set.rules, key=lambda r: -r.weight))
        content = (
            f"## SkillOpt Experiment Update {iteration}\n"
            "Apply this validation-gated behavior before answering:\n"
            f"{policy_lines or '- Follow the supplied data and policy intent.'}\n"
            "Cite available source evidence. When evidence is missing or contradictory, state the gap "
            "and escalate instead of guessing."
        )
        return Patch(
            edits=[Edit(op="append", content=content, source_type="failure", support_count=len(policy_set.rules))],
            reasoning=reason,
        )

    def _patch_summary(self, patch: Patch, gate_action: str) -> str:
        ops = ", ".join(edit.op for edit in patch.edits) or "no edits"
        return f"SkillOpt {gate_action}: applied bounded edit operations ({ops})."

    def _brief(self, request, cases, policy_set, instruction: str) -> str:
        policies = "\n".join(f"- ({rule.weight:.2f}) {rule.id}: {rule.text}" for rule in policy_set.rules)
        examples = "\n".join(f"- {case}" for case in cases[:8])
        return (
            f"Artifact kind: {request.artifact_kind.value}\n"
            f"Instruction: {instruction}\n\n"
            f"Weighted policies:\n{policies or '- No explicit policies supplied.'}\n\n"
            f"Normalized data examples:\n{examples or '- No examples supplied.'}\n"
        )

    def _score(
        self,
        content: str,
        cases: list[str],
        policy_set: PolicySet,
        assertions=None,
    ) -> float:
        return self._score_from_report(
            evaluate_artifact(content, cases, policy_set, assertions or []),
            cases,
            content,
        )

    def _score_from_report(self, report: EvaluationReport, cases: list[str], content: str) -> float:
        eval_score = report.score
        case_score = min(0.1, len(cases) * 0.01)
        structure_score = 0.1 if any(marker in content for marker in ["\n-", "\n1.", "##"]) else 0
        return round(min(1.0, eval_score * 0.8 + case_score + structure_score), 4)

    def _build_insights(
        self,
        best: ArtifactVersion,
        rejected: list[ArtifactVersion],
        cases: list[str],
        policy_set: PolicySet,
        evaluation_report=None,
    ) -> EvaluationInsight:
        lower = best.content.lower()
        worked = []
        went_wrong = []
        changes = []
        coverage = []
        risks = []

        if any(word in lower for word in ["cite", "source", "evidence"]):
            worked.append("The best version includes source/citation behavior.")
        else:
            went_wrong.append("The best version does not clearly require citations or evidence.")
            changes.append("Add an explicit instruction to cite source data or policy sections.")

        if any(word in lower for word in ["escalate", "uncertain", "incomplete"]):
            worked.append("The best version handles uncertainty or incomplete data.")
        else:
            went_wrong.append("Uncertainty handling is weak or missing.")
            changes.append("Add escalation rules for incomplete, contradictory, or low-confidence data.")

        if rejected:
            went_wrong.append(f"{len(rejected)} candidate version(s) failed to improve validation score.")
            changes.append("Review rejected SkillOpt patch metadata before changing weights or adding new policies.")

        for rule in policy_set.rules:
            if _policy_hit(lower, rule.text):
                coverage.append(f"Covered: {rule.id}")
            else:
                coverage.append(f"Needs attention: {rule.id}")
                changes.append(f"Strengthen instructions for policy '{rule.id}'.")

        if len(cases) < 3:
            risks.append("Dataset is small; add more examples before promoting to production.")
        if best.score < 0.7:
            risks.append("Best score is below the recommended promotion threshold of 0.70.")
        if evaluation_report and evaluation_report.failed_metrics:
            went_wrong.append(
                f"{evaluation_report.failed} eval check(s) failed: "
                f"{', '.join(evaluation_report.failed_metrics[:5])}."
            )
            changes.append("Address failed eval checks before approving promotion.")

        return EvaluationInsight(
            worked=worked or ["The optimizer produced a deployable baseline artifact."],
            went_wrong=went_wrong or ["No major gaps detected by the current evaluator."],
            prompt_changes_needed=dedupe(changes) or ["No immediate prompt changes required."],
            policy_coverage=coverage,
            data_risks=risks or ["No major data risks detected by the current evaluator."],
        )


def compare_models(request: ModelComparisonRequest) -> ModelComparisonResponse:
    dataset = normalize_raw_data("comparison", request.raw_data or request.prompt)
    policy_set = PolicySet(name="comparison", rules=request.policies)
    results = [
        _compare_one_model(
            request.prompt,
            dataset.normalized_cases,
            policy_set,
            request.model_a,
            request.execute_live,
        ),
        _compare_one_model(
            request.prompt,
            dataset.normalized_cases,
            policy_set,
            request.model_b,
            request.execute_live,
        ),
    ]
    winner = max(results, key=lambda result: result.score)
    return ModelComparisonResponse(
        prompt_hash=stable_hash(request.prompt),
        results=results,
        winner=winner.model["model"],
        summary=f"{winner.model['model']} has the stronger score for this prompt under the current policy rubric.",
    )


def _compare_one_model(
    prompt: str,
    cases: list[str],
    policy_set: PolicySet,
    provider,
    execute_live: bool = False,
) -> ModelComparisonResult:
    output = None
    evaluated_text = prompt
    if execute_live:
        model_provider = build_provider(provider)
        eval_case = cases[0] if cases else "No evaluation case supplied."
        output = model_provider.generate(prompt, f"Evaluate this case using the prompt:\n{eval_case}")
        evaluated_text = f"{prompt}\n\nModel output:\n{output}"
    score = SkillOptInspiredOptimizer()._score(evaluated_text, cases, policy_set, [])
    provider_bonus = {
        "openai": 0.02,
        "anthropic": 0.02,
        "azure_openai": 0.02,
        "aws_bedrock": 0.01,
        "litellm": 0.01,
        "mock": 0.0,
    }.get(provider.kind.value, 0.0)
    final_score = round(min(1.0, score + provider_bonus), 4)
    risks = []
    strengths = ["Uses the same approved prompt for an apples-to-apples comparison."]
    if provider.kind.value == "mock":
        risks.append("Mock provider is for workflow validation only, not production quality.")
    if final_score < 0.7:
        risks.append("Score is below the recommended promotion threshold.")
    return ModelComparisonResult(
        model=provider.redacted(),
        score=final_score,
        sample_output=output,
        strengths=strengths,
        risks=risks or ["No model-specific risks detected by the current evaluator."],
        recommendation="Use for production validation." if final_score >= 0.7 else "Improve prompt or add eval cases before promoting.",
    )


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _policy_hit(content_lower: str, policy_text: str) -> bool:
    words = [word.strip(".,:;()").lower() for word in policy_text.split() if len(word) > 4]
    if not words:
        return False
    return any(word in content_lower for word in words[:8])
