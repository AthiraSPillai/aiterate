from __future__ import annotations

import json

from aiterate.domain import (
    BehaviorEvaluationReport,
    EvalAssertion,
    EvalCase,
    PolicySet,
    TargetExecutionResult,
)
from aiterate.evaluator import evaluate_artifact
from aiterate.providers.base import ModelProvider


TARGET_SYSTEM = """You are the target model being evaluated by AIterate.
Follow the candidate prompt exactly. Answer only the eval case."""


def build_eval_cases(normalized_cases: list[str]) -> list[EvalCase]:
    return [_case_from_text(index, case) for index, case in enumerate(normalized_cases, start=1)]


def evaluate_candidate_behavior(
    candidate_prompt: str,
    cases: list[EvalCase],
    policy_set: PolicySet,
    target_provider: ModelProvider,
    assertions: list[EvalAssertion],
) -> BehaviorEvaluationReport:
    executions = []
    for case in cases:
        output = target_provider.generate(
            TARGET_SYSTEM,
            f"Candidate prompt:\n{candidate_prompt}\n\nEval input:\n{case.input}",
        )
        report = evaluate_artifact(
            output,
            [case.input, *( [case.expected] if case.expected else [] )],
            policy_set,
            _case_assertions(case, assertions),
        )
        executions.append(
            TargetExecutionResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected,
                output=output,
                score=report.score,
                failed_metrics=report.failed_metrics,
            )
        )
    if not executions:
        return BehaviorEvaluationReport(score=0, pass_rate=0, case_count=0)
    score = sum(execution.score for execution in executions) / len(executions)
    passed = sum(1 for execution in executions if execution.score >= 0.7)
    failed_metrics = list(dict.fromkeys(metric for execution in executions for metric in execution.failed_metrics))
    return BehaviorEvaluationReport(
        score=round(score, 4),
        pass_rate=round(passed / len(executions), 4),
        case_count=len(executions),
        executions=executions,
        failed_metrics=failed_metrics,
    )


def _case_assertions(case: EvalCase, assertions: list[EvalAssertion]) -> list[EvalAssertion]:
    if not case.expected:
        return assertions
    from aiterate.domain import AssertionKind

    return [
        *assertions,
        EvalAssertion(
            id=f"{case.id}_expected_similarity",
            type=AssertionKind.SEMANTIC_SIMILARITY,
            value=case.expected,
            threshold=0.25,
            metric="expected_similarity",
            weight=1,
            description="Target output should overlap with the expected answer.",
        ),
    ]


def _case_from_text(index: int, text: str) -> EvalCase:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return EvalCase(id=f"case_{index}", input=text)
    if not isinstance(payload, dict):
        return EvalCase(id=f"case_{index}", input=text)
    input_text = (
        payload.get("input")
        or payload.get("question")
        or payload.get("prompt")
        or payload.get("case")
        or json.dumps(payload, sort_keys=True)
    )
    expected = payload.get("expected") or payload.get("answer") or payload.get("output")
    tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
    return EvalCase(
        id=str(payload.get("id") or f"case_{index}"),
        input=str(input_text),
        expected=str(expected) if expected is not None else None,
        tags=[str(tag) for tag in tags],
        metadata={key: value for key, value in payload.items() if key not in {"id", "input", "question", "prompt", "case", "expected", "answer", "output", "tags"}},
    )
