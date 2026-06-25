from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from aiterate.config import settings
from aiterate.domain import ArtifactVersion, OptimizationRun


class GitVersionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.storage_dir / "repo"
        self.artifacts_dir = self.root / "artifacts"

    def init(self) -> None:
        if not settings.enable_local_git:
            return
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not (self.root / ".git").exists():
            self._git("init")
            self._git("config", "user.name", settings.git_author_name)
            self._git("config", "user.email", settings.git_author_email)

    def commit_version(self, run: OptimizationRun, version: ArtifactVersion) -> str:
        if not settings.enable_local_git:
            return f"db://{run.artifact_id}/v{version.version}"
        self.init()
        artifact_dir = self.artifacts_dir / run.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        content_path = artifact_dir / f"v{version.version}.{version.kind.value}.md"
        metadata_path = artifact_dir / f"v{version.version}.metadata.json"
        run_path = artifact_dir / "run.json"
        content_path.write_text(version.content, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(version.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        run_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8")
        self._git(
            "add",
            content_path.relative_to(self.root).as_posix(),
            metadata_path.relative_to(self.root).as_posix(),
            run_path.relative_to(self.root).as_posix(),
        )
        self._git(
            "commit",
            "-m",
            f"aiterate: {run.name} v{version.version} score={version.score:.3f}",
        )
        self._git("tag", "-f", f"{run.artifact_id}-v{version.version}")
        return self._git("rev-parse", "HEAD").strip()

    def _git(self, *args: str) -> str:
        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"] = settings.git_author_name
        env["GIT_AUTHOR_EMAIL"] = settings.git_author_email
        env["GIT_COMMITTER_NAME"] = settings.git_author_name
        env["GIT_COMMITTER_EMAIL"] = settings.git_author_email
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=True,
            env=env,
        )
        return result.stdout
