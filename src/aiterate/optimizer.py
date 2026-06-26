from __future__ import annotations

import json
import random

from skillopt.evaluation.gate import evaluate_gate
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.types import Edit, Patch

from aiterate.behavior import build_eval_cases, evaluate_candidate_behavior
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
    PriorityRule,
    new_id,
)
from aiterate.evaluator import evaluate_artifact
from aiterate.ingest import normalize_raw_data, stable_hash
from aiterate.pricing import RunCostMeter
from aiterate.providers import build_provider
from aiterate.tracking import build_tracker
from aiterate.versioning import GitVersionStore


SYSTEM_PROMPT = """You are AIterate's artifact optimizer.
Create production-ready AI prompts or agent skill documents from examples, weighted policies, and knowledge-base references.
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
        train_cases, validation_cases = self._split_cases(dataset.normalized_cases, request.validation_split, request.seed)
        policy_set = self._policy_set(request)
        provider = build_provider(request.provider)
        cost_meter = RunCostMeter(request.provider.kind.value, request.provider.model)
        tracker = build_tracker(
            request.tracker,
            tracker_uri=request.tracker_uri,
            api_key=request.tracker_api_key.get_secret_value() if request.tracker_api_key else None,
            project=request.tracker_project,
        )
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
                "iterations": request.iterations,
                "requested_iterations": request.iterations,
                "validation_split": request.validation_split,
                "promotion_threshold": request.promotion_threshold,
                "max_budget_usd": request.max_budget_usd,
                "run_target_validation": request.run_target_validation,
                "train_case_count": len(train_cases),
                "validation_case_count": len(validation_cases),
                "context_roles": {
                    "data_examples": "train_test_cases",
                    "policy_context": "weighted_scoring_rules",
                    "knowledge_base_context": "grounding_references",
                },
                "policy_context_hash": stable_hash(request.policy_context or ""),
                "knowledge_base_hash": stable_hash(request.knowledge_base_context or ""),
                "policy_context": request.policy_context or "",
                "knowledge_base_context": request.knowledge_base_context or "",
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
                "tracker_uri": request.tracker_uri or "",
                "tracker_project": request.tracker_project or "",
                "git_tracking": str(request.enable_git_tracking),
                "create_pull_request": str(request.create_pull_request),
                "dataset_hash": dataset.content_hash,
                "policy_hash": policy_hash,
                "validation_split": str(request.validation_split),
                "promotion_threshold": str(request.promotion_threshold),
                "max_budget_usd": "" if request.max_budget_usd is None else str(request.max_budget_usd),
                "run_target_validation": str(request.run_target_validation),
                "train_case_count": str(len(train_cases)),
                "validation_case_count": str(len(validation_cases)),
                "policy_context_hash": stable_hash(request.policy_context or ""),
                "knowledge_base_hash": stable_hash(request.knowledge_base_context or ""),
            },
        )
        if getattr(tracker, "last_error", None):
            run.optimizer["tracking_error"] = tracker.last_error
        try:
            best = self._initial_version(request, train_cases, validation_cases, policy_set, provider, cost_meter)
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
                        "train_case_count": len(train_cases),
                        "validation_case_count": len(validation_cases),
                        "context_roles": {
                            "data_examples": "train_test_cases",
                            "policy_context": "weighted_scoring_rules",
                            "knowledge_base_context": "grounding_references",
                        },
                    },
                }
            )
            run.accepted_versions.append(best)
            self.version_store.commit_version(run, best)
            tracker.log_metric("score", best.score, step=0)
            if getattr(tracker, "last_error", None):
                run.optimizer["tracking_error"] = tracker.last_error

            for iteration in range(1, request.iterations + 1):
                current_cost = cost_meter.estimate().total_cost
                if request.max_budget_usd is not None and current_cost >= request.max_budget_usd:
                    run.optimizer["budget_stop_reason"] = (
                        f"Stopped before iteration {iteration}; estimated cost "
                        f"{current_cost:.6f} reached the configured cap "
                        f"{request.max_budget_usd:.6f}."
                    )
                    break
                candidate = self._candidate_version(
                    request,
                    best,
                    train_cases,
                    validation_cases,
                    policy_set,
                    provider,
                    cost_meter,
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
                if getattr(tracker, "last_error", None):
                    run.optimizer["tracking_error"] = tracker.last_error
            run.best_version = best
            run.evaluation_report = evaluate_artifact(
                best.content,
                validation_cases,
                policy_set,
                request.eval_assertions,
            )
            if request.run_target_validation:
                try:
                    target_config = request.target_provider or request.provider
                    target_provider = build_provider(target_config)
                    run.behavior_report = evaluate_candidate_behavior(
                        best.content,
                        build_eval_cases(validation_cases),
                        policy_set,
                        target_provider,
                        request.eval_assertions,
                    )
                    best.behavior_report = run.behavior_report
                    tracker.log_metric("target_validation_score", run.behavior_report.score, step=request.iterations)
                    tracker.log_metric("target_validation_pass_rate", run.behavior_report.pass_rate, step=request.iterations)
                except Exception as exc:
                    run.optimizer["target_validation_error"] = f"Target model test failed: {exc}"
            run.insights = self._build_insights(
                request,
                best,
                run.rejected_versions,
                validation_cases,
                policy_set,
                run.evaluation_report,
            )
            run.cost_estimate = cost_meter.estimate()
            tracker.log_metric("estimated_cost", run.cost_estimate.total_cost, step=request.iterations)
            if getattr(tracker, "last_error", None):
                run.optimizer["tracking_error"] = tracker.last_error
            return run
        finally:
            tracker.end_run()

    def _initial_version(
        self,
        request,
        train_cases,
        validation_cases,
        policy_set,
        provider,
        cost_meter,
    ) -> ArtifactVersion:
        if request.baseline_artifact and request.baseline_artifact.strip():
            content = request.baseline_artifact.strip()
            change_summary = "User baseline artifact imported as the optimization starting point."
        else:
            user = self._brief(request, train_cases, policy_set, "Create the initial baseline artifact.")
            content = self._generate(provider, cost_meter, SYSTEM_PROMPT, user)
            change_summary = "Initial artifact generated from raw data and weighted policies."
        score = self._score(content, validation_cases, policy_set, request.eval_assertions)
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
            metadata={"skillopt_stage": "baseline", "scored_on": "validation_holdout"},
        )

    def _candidate_version(
        self,
        request,
        best,
        train_cases,
        validation_cases,
        policy_set,
        provider,
        cost_meter,
        iteration,
    ) -> ArtifactVersion:
        before_report = evaluate_artifact(best.content, validation_cases, policy_set, request.eval_assertions)
        patch = self._candidate_patch(
            request,
            best,
            train_cases,
            policy_set,
            provider,
            cost_meter,
            iteration,
            before_report,
        )
        content, patch_report = apply_patch_with_report(best.content, patch)
        if content.strip() == best.content.strip():
            patch = self._fallback_patch(policy_set, iteration, reason="provider patch produced no text change")
            content, patch_report = apply_patch_with_report(best.content, patch)

        after_report = evaluate_artifact(content, validation_cases, policy_set, request.eval_assertions)
        score = self._score_from_report(after_report, validation_cases, content)
        score_delta = round(score - best.score, 4)
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
                "score_delta": score_delta,
                "candidate_eval_score": after_report.score,
                "gate_rule": "candidate_score > current_best_score",
                "gate_reason": _gate_reason(score_delta, score, best.score),
                "failed_metrics": after_report.failed_metrics,
                "scored_on": "validation_holdout",
                "train_case_count": len(train_cases),
                "validation_case_count": len(validation_cases),
            },
        )

    def _candidate_patch(
        self,
        request: OptimizationRequest,
        best: ArtifactVersion,
        cases: list[str],
        policy_set: PolicySet,
        provider,
        cost_meter,
        iteration: int,
        before_report: EvaluationReport,
    ) -> Patch:
        user = (
            f"{self._brief(request, cases, policy_set, f'Reflect on version {best.version} and propose bounded edits.')}\n"
            f"Current artifact:\n{best.content}\n\n"
            f"Current validation score: {best.score}\n"
            f"Failed metrics: {', '.join(before_report.failed_metrics) or 'none'}\n"
            "Use small add/delete/replace-style edits. Prefer measurable changes."
        )
        raw = self._generate(provider, cost_meter, SKILLOPT_SYSTEM_PROMPT, user)
        patch = self._parse_patch(raw)
        if patch.edits:
            return patch
        return self._fallback_patch(policy_set, iteration, reason="provider did not return parseable text edits")

    def _policy_set(self, request: OptimizationRequest) -> PolicySet:
        rules = [*request.policies]
        derived = self._policy_rules_from_context(
            request.policy_context or "",
            existing_ids={rule.id for rule in rules},
            existing_texts={_normalize_policy_text(rule.text) for rule in rules},
        )
        return PolicySet(name=request.name, rules=[*rules, *derived])

    def _policy_rules_from_context(
        self,
        policy_context: str,
        existing_ids: set[str],
        existing_texts: set[str] | None = None,
    ) -> list[PriorityRule]:
        existing_texts = existing_texts or set()
        chunks = [
            chunk.strip(" -\t")
            for chunk in policy_context.splitlines()
            if len(chunk.strip(" -\t")) > 12
        ]
        rules = []
        for index, chunk in enumerate(chunks[:8], start=1):
            normalized = _normalize_policy_text(chunk)
            if normalized in existing_texts:
                continue
            rule_id = f"policy_file_{index}"
            if rule_id in existing_ids:
                continue
            rules.append(PriorityRule(id=rule_id, text=chunk[:500], weight=0.15))
            existing_texts.add(normalized)
        return rules

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

    def _generate(self, provider, cost_meter: RunCostMeter, system: str, user: str) -> str:
        output = provider.generate(system, user)
        cost_meter.record(system, user, output)
        return output

    def _fallback_patch(self, policy_set: PolicySet, iteration: int, reason: str) -> Patch:
        policy_lines = "\n".join(f"- {rule.text}" for rule in sorted(policy_set.rules, key=lambda r: -r.weight))
        content = (
            f"## Aiterate Candidate Update {iteration}\n"
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
        operation_labels = {
            "append": "added new guidance",
            "insert_after": "inserted guidance into the existing artifact",
            "replace": "rewrote part of the artifact",
            "delete": "removed text",
        }
        ops = [operation_labels.get(edit.op, edit.op.replace("_", " ")) for edit in patch.edits]
        action = {
            "accept_new_best": "Promoted as the new best version",
            "accept": "Accepted as an improvement",
            "reject": "Not promoted",
        }.get(gate_action, gate_action.replace("_", " ").title())
        if not ops:
            return f"{action}: no text edit was proposed."
        unique_ops = ", ".join(dedupe(ops))
        return f"{action}: {unique_ops}."

    def _split_cases(self, cases: list[str], validation_split: float, seed: int) -> tuple[list[str], list[str]]:
        if len(cases) <= 1:
            return cases, cases
        indices = list(range(len(cases)))
        random.Random(seed).shuffle(indices)
        validation_count = min(len(cases) - 1, max(1, round(len(cases) * validation_split)))
        validation_indices = set(indices[:validation_count])
        train_cases = [case for index, case in enumerate(cases) if index not in validation_indices]
        validation_cases = [case for index, case in enumerate(cases) if index in validation_indices]
        return train_cases or cases, validation_cases or cases

    def _brief(self, request, cases, policy_set, instruction: str) -> str:
        policies = "\n".join(f"- ({rule.weight:.2f}) {rule.id}: {rule.text}" for rule in policy_set.rules)
        examples = "\n".join(f"- {case}" for case in cases[:8])
        policy_context = _excerpt(request.policy_context or "")
        knowledge_base_context = _excerpt(request.knowledge_base_context or "")
        return (
            f"Artifact kind: {request.artifact_kind.value}\n"
            f"Instruction: {instruction}\n\n"
            "Context roles:\n"
            "- Data / Examples: train/test cases for optimization and regression scoring.\n"
            "- Policies: weighted scoring rules and acceptance criteria.\n"
            "- Knowledge Base / References: grounding material to cite or follow, not eval cases.\n\n"
            f"Weighted policies:\n{policies or '- No explicit policies supplied.'}\n\n"
            f"Policy file context:\n{policy_context or '- No policy file context supplied.'}\n\n"
            f"Knowledge Base / References:\n{knowledge_base_context or '- No knowledge base supplied.'}\n\n"
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
        request: OptimizationRequest,
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
            changes.append("Review rejected candidate metadata before changing weights or adding new policies.")

        for rule in policy_set.rules:
            if _policy_hit(lower, rule.text):
                coverage.append(f"Covered: {rule.id}")
            else:
                coverage.append(f"Needs attention: {rule.id}")
                changes.append(f"Strengthen instructions for policy '{rule.id}'.")

        if len(cases) < 3:
            risks.append("Dataset is small; add more examples before promoting to production.")
        if best.score < request.promotion_threshold:
            risks.append(
                f"Best score is below the configured promotion threshold of "
                f"{request.promotion_threshold:.2f}."
            )
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
    best_score = max(result.score for result in results)
    winners = [result for result in results if result.score == best_score]
    winner = winners[0]
    summary = (
        "The selected models tied under the current comparison rubric."
        if len(winners) > 1
        else f"{winner.model['model']} has the stronger score for this prompt under the current comparison rubric."
    )
    return ModelComparisonResponse(
        prompt_hash=stable_hash(request.prompt),
        results=results,
        winner=winner.model["model"],
        summary=summary,
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
    prompt_score = SkillOptInspiredOptimizer()._score(evaluated_text, cases, policy_set, [])
    model_score = _model_fit_score(provider.kind.value, provider.model)
    if execute_live:
        final_score = round(min(1.0, prompt_score * 0.92 + model_score * 0.08), 4)
    else:
        final_score = round(min(1.0, prompt_score * 0.82 + model_score * 0.18), 4)
    risks = _model_fit_risks(provider.kind.value, provider.model, execute_live)
    strengths = [
        "Uses the same approved prompt for an apples-to-apples comparison.",
        (
            "Live score includes target-model output on a holdout example."
            if execute_live
            else "Offline score estimates model fit from the prompt, policy rubric, and selected model profile."
        ),
    ]
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


def _model_fit_score(provider: str, model: str) -> float:
    model_lower = (model or "").lower()
    if provider == "mock":
        return 0.5
    if any(token in model_lower for token in ["gpt-5.5", "gpt-5", "opus", "sonnet-4", "claude-4"]):
        return 0.96
    if any(token in model_lower for token in ["gpt-5.4", "gpt-4.1", "sonnet", "gemini-1.5-pro", "gemini-pro"]):
        return 0.91
    if any(token in model_lower for token in ["gpt-4o", "llama-3.1-70b", "mistral-large"]):
        return 0.87
    if any(token in model_lower for token in ["mini", "haiku", "flash", "small", "8b"]):
        return 0.78
    if any(token in model_lower for token in ["local", "ollama", "openai-compatible"]):
        return 0.72
    return {
        "openai": 0.84,
        "azure_openai": 0.84,
        "anthropic": 0.84,
        "aws_bedrock": 0.82,
        "litellm": 0.78,
    }.get(provider, 0.74)


def _model_fit_risks(provider: str, model: str, execute_live: bool) -> list[str]:
    model_lower = (model or "").lower()
    risks = []
    if not execute_live:
        risks.append("Offline comparison did not call the model; enable live eval before final provider selection.")
    if any(token in model_lower for token in ["mini", "haiku", "flash", "small", "8b"]):
        risks.append("Smaller or lower-cost models may need stricter regression tests for complex policies.")
    if provider in {"litellm", "aws_bedrock"}:
        risks.append("Verify provider-specific model ID, region, and quota before production use.")
    if any(token in model_lower for token in ["local", "ollama", "openai-compatible"]):
        risks.append("Local or OpenAI-compatible APIs can vary by deployment; run live validation before approval.")
    return risks


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _normalize_policy_text(text: str) -> str:
    return " ".join(text.lower().split())


def _gate_reason(score_delta: float, candidate_score: float, current_score: float) -> str:
    if score_delta > 0:
        return (
            f"Accepted because candidate score improved by {score_delta:.4f} "
            f"from {current_score:.4f} to {candidate_score:.4f}."
        )
    if score_delta == 0:
        return (
            f"Rejected because candidate score tied the current best at {current_score:.4f}; "
            "the gate only promotes measurable improvements."
        )
    return (
        f"Rejected because candidate score was {abs(score_delta):.4f} below the current best "
        f"({candidate_score:.4f} vs {current_score:.4f})."
    )


def _excerpt(text: str, limit: int = 2400) -> str:
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit].rstrip()}\n...[truncated]"


def _policy_hit(content_lower: str, policy_text: str) -> bool:
    words = [word.strip(".,:;()").lower() for word in policy_text.split() if len(word) > 4]
    if not words:
        return False
    return any(word in content_lower for word in words[:8])
