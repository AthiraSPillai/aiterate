from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich import print

from aiterate.domain import (
    ArtifactKind,
    AssertionKind,
    EvalAssertion,
    OptimizationRequest,
    PolicySet,
    PriorityRule,
    ProviderConfig,
    ProviderKind,
)
from aiterate.evaluator import evaluate_artifact
from aiterate.optimizer import SkillOptInspiredOptimizer

app = typer.Typer(help="AIterate: create, optimize, version, and trace AI prompts and skills.")


@app.command()
def init() -> None:
    """Initialize the local Git-backed artifact store."""
    from aiterate.versioning import GitVersionStore

    GitVersionStore().init()
    print("[green]Initialized .aiterate repo[/green]")


@app.command()
def migrate() -> None:
    """Apply Alembic database migrations."""
    from aiterate.db import run_migrations

    run_migrations()
    print("[green]Database migrations applied[/green]")


@app.command()
def worker(once: bool = typer.Option(False, help="Run one queued job and exit.")) -> None:
    """Run queued optimizer jobs."""
    import time

    from aiterate.jobs import run_one_optimization_job

    while True:
        job = run_one_optimization_job()
        if job:
            print(f"[green]Processed job[/green] {job.id}: {job.status.value}")
        elif once:
            print("[yellow]No queued jobs[/yellow]")
            return
        if once:
            return
        time.sleep(2)


@app.command()
def optimize(
    name: str = typer.Option(..., help="Project or artifact name."),
    data: Path = typer.Option(..., exists=True, readable=True, help="Raw data file."),
    baseline: Path | None = typer.Option(None, exists=True, readable=True, help="Optional existing prompt or skill file."),
    policy: Path | None = typer.Option(None, exists=True, readable=True, help="Simple policy file."),
    kind: ArtifactKind = ArtifactKind.PROMPT,
    provider: ProviderKind = ProviderKind.MOCK,
    model: str = "mock-optimizer",
    iterations: int = 3,
) -> None:
    """Create and optimize a prompt or skill artifact."""
    raw_data = data.read_text(encoding="utf-8")
    baseline_artifact = baseline.read_text(encoding="utf-8") if baseline else None
    policies = _load_policies(policy)
    request = OptimizationRequest(
        name=name,
        artifact_kind=kind,
        raw_data=raw_data,
        baseline_artifact=baseline_artifact,
        policies=policies,
        provider=ProviderConfig(kind=provider, model=model),
        iterations=iterations,
    )
    run = SkillOptInspiredOptimizer().optimize(request)
    print(f"[green]Best score:[/green] {run.best_version.score if run.best_version else 0}")
    print(f"[green]Artifact:[/green] {run.artifact_id}")
    if run.best_version:
        print(run.best_version.content)


@app.command("eval")
def eval_artifact(
    artifact: Path = typer.Option(..., exists=True, readable=True, help="Prompt or skill file to evaluate."),
    data: Path = typer.Option(..., exists=True, readable=True, help="Eval data file."),
    policy: Path | None = typer.Option(None, exists=True, readable=True, help="Policy file."),
    assertion: Path | None = typer.Option(None, exists=True, readable=True, help="Optional assertions JSON/YAML file."),
    min_score: float = typer.Option(0.7, min=0, max=1, help="Minimum passing score."),
) -> None:
    """Evaluate a prompt or skill artifact with AIterate native checks."""
    from aiterate.ingest import normalize_raw_data

    content = artifact.read_text(encoding="utf-8")
    raw_data = data.read_text(encoding="utf-8")
    dataset = normalize_raw_data("cli-eval", raw_data)
    policies = _load_policies(policy)
    assertions = _load_assertions(assertion)
    report = evaluate_artifact(content, dataset.normalized_cases, PolicySet(rules=policies), assertions)

    print(f"[green]Eval score:[/green] {report.score}")
    print(f"[green]Pass rate:[/green] {report.pass_rate:.0%}")
    for check in report.checks:
        status = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        print(f"{status} {check.metric}: {check.message}")
    if report.score < min_score:
        raise typer.Exit(1)


def _load_policies(path: Path | None) -> list[PriorityRule]:
    if not path:
        return []
    text = path.read_text(encoding="utf-8")
    structured = _load_structured_policies(text)
    if structured:
        return structured
    rules: list[PriorityRule] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip(" -")
        if not stripped or stripped.startswith(("policies:", "id:", "text:", "weight:")):
            continue
        rules.append(PriorityRule(id=f"policy_{idx}", text=stripped, weight=1))
    return rules


def _load_structured_policies(text: str) -> list[PriorityRule]:
    payload = _parse_json_or_yaml(text)
    if not payload:
        return []
    rows = payload.get("policies") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    rules = []
    for idx, row in enumerate(rows, start=1):
        if isinstance(row, str):
            rules.append(PriorityRule(id=f"policy_{idx}", text=row, weight=1))
        elif isinstance(row, dict):
            rules.append(
                PriorityRule(
                    id=str(row.get("id") or f"policy_{idx}"),
                    text=str(row.get("text") or row.get("description") or ""),
                    weight=float(row.get("weight", 1)),
                )
            )
    return [rule for rule in rules if rule.text]


def _load_assertions(path: Path | None) -> list[EvalAssertion]:
    if not path:
        return []
    payload = _parse_json_or_yaml(path.read_text(encoding="utf-8"))
    if not payload:
        return []
    rows = payload.get("assertions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    assertions = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        assertions.append(
            EvalAssertion(
                id=str(row.get("id") or f"assert_{idx}"),
                type=AssertionKind(str(row.get("type") or "contains")),
                value=row.get("value"),
                threshold=row.get("threshold"),
                weight=float(row.get("weight", 1)),
                metric=row.get("metric"),
                description=str(row.get("description") or ""),
            )
        )
    return assertions


def _parse_json_or_yaml(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import yaml
    except ImportError:
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


if __name__ == "__main__":
    app()
