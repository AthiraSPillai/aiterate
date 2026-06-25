from pathlib import Path

from typer.testing import CliRunner

from aiterate.cli import _load_policies, app

runner = CliRunner()


def test_load_yaml_policies(tmp_path: Path):
    path = tmp_path / "policies.yml"
    path.write_text(
        """
policies:
  - id: cite
    text: Always cite sources.
    weight: 0.7
  - id: escalate
    text: Escalate incomplete data.
    weight: 0.3
""",
        encoding="utf-8",
    )

    policies = _load_policies(path)

    assert [policy.id for policy in policies] == ["cite", "escalate"]
    assert policies[0].weight == 0.7


def test_eval_command_passes_for_grounded_artifact(tmp_path: Path):
    artifact = tmp_path / "prompt.md"
    data = tmp_path / "data.txt"
    policy = tmp_path / "policies.yml"
    artifact.write_text(
        "Always cite source evidence and escalate when data is incomplete.",
        encoding="utf-8",
    )
    data.write_text("Customer support answers need citations and uncertainty escalation.", encoding="utf-8")
    policy.write_text(
        """
policies:
  - id: cite
    text: Always cite source evidence.
    weight: 0.5
  - id: escalate
    text: Escalate incomplete data.
    weight: 0.5
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "--artifact", str(artifact), "--data", str(data), "--policy", str(policy)])

    assert result.exit_code == 0
    assert "Eval score" in result.output
