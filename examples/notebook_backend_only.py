"""Backend-only AIterate example for notebooks or Python scripts.

Run with:
    python examples/notebook_backend_only.py
"""

from pathlib import Path

from aiterate.domain import OptimizationRequest, PriorityRule, ProviderConfig, ProviderKind
from aiterate.sdk import AIterateClient


raw_data = Path("examples/raw_support_notes.txt").read_text(encoding="utf-8")

policies = [
    PriorityRule(
        id="cite_sources",
        text="Always cite the policy or dataset section used to answer.",
        weight=0.35,
    ),
    PriorityRule(
        id="concise",
        text="Keep answers under 180 words unless the user asks for detail.",
        weight=0.20,
    ),
    PriorityRule(
        id="escalate_uncertainty",
        text="Escalate to a human reviewer when source data is incomplete or contradictory.",
        weight=0.45,
    ),
]

client = AIterateClient()
run = client.optimize(
    OptimizationRequest(
        name="support-agent-notebook",
        raw_data=raw_data,
        policies=policies,
        provider=ProviderConfig(kind=ProviderKind.MOCK, model="mock-optimizer"),
        iterations=3,
    )
)

print(f"Artifact: {run.artifact_id}")
print(f"Best score: {run.best_version.score if run.best_version else 0}")
print(run.best_version.content if run.best_version else "No version created")
